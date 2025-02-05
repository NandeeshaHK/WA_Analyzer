import zipfile
import os
import re
import emoji
import time
import logging
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from datetime import datetime
from collections import defaultdict
from groq import Groq

class EnhancedChunkedChatAnalyzer:
    def __init__(self, chunk_size=50, delay_between_requests=2, response_threshold_minutes=60):
        # Setup logging
        os.makedirs('debug', exist_ok=True)
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s',
            handlers=[
                logging.FileHandler('debug/chat_analyzer.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        # Download necessary NLTK resources
        nltk.download('stopwords', quiet=True)
        
        self.chat_content = ""
        self.processed_messages = []
        self.input_tokens = 0
        self.output_tokens = 0
        self.chunk_size = chunk_size
        self.delay = delay_between_requests
        self.participants = []
        self.response_threshold = response_threshold_minutes
        self.client = Groq()
        self.full_conversation_context = []
        
        # Text preprocessing tools
        self.stopwords = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()

    def save_conversation_context(self, conversation_context, file_path=None):
        """
        Save conversation context to a pickle file
        
        Args:
        - conversation_context (list): The conversation context to save
        - file_path (str, optional): Custom file path. If None, uses a default path
        
        Returns:
        - str: Path where the context was saved
        """
        # Create debug directory if it doesn't exist
        os.makedirs('debug', exist_ok=True)
        
        # If no file path provided, generate a default one
        if file_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f'debug/conversation_context_{timestamp}.pickle'
        
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(conversation_context, f)
            
            self.logger.info(f"Conversation context saved to {file_path}")
            self.conversation_context_path = file_path
            return file_path
        
        except Exception as e:
            self.logger.error(f"Failed to save conversation context: {e}")
            return None

    def load_conversation_context(self, file_path=None):
        """
        Load conversation context from a pickle file
        
        Args:
        - file_path (str, optional): Path to the pickle file. 
          If None, uses the last saved context path
        
        Returns:
        - list or None: Loaded conversation context
        """
        # If no path provided, use the last saved path
        if file_path is None:
            file_path = self.conversation_context_path
        
        # If still no path, return None
        if not file_path or not os.path.exists(file_path):
            self.logger.warning("No saved conversation context found")
            return None
        
        try:
            with open(file_path, 'rb') as f:
                conversation_context = pickle.load(f)
            
            self.logger.info(f"Conversation context loaded from {file_path}")
            return conversation_context
        
        except Exception as e:
            self.logger.error(f"Failed to load conversation context: {e}")
            return None
        
    def process_message(self, content):
        """Process raw chat content into structured format with emoji removal"""
        messages = []
        # Updated pattern to handle Unicode spaces and make space matching more flexible
        pattern = r'(\d{2}/\d{2}/\d{4},\s+\d{1,2}:\d{2}\s*[ap]m)\s*-\s*([^:]+):\s*(.+)'
        
        # Take only first 25 lines for name extraction
        lines = content.splitlines()[:25]
        content_sample = '\n'.join(lines)
        
        # Query model to extract names
        client = Groq()
        name_prompt = f"""Given these first few lines of a WhatsApp chat, extract only the unique participant names (excluding system messages):

    {content_sample}

    Return only the names, one per line."""
        
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helper that extracts participant names from chat logs. Return only the names, one per line. There will be max two active chat participants"
                },
                {
                    "role": "user",
                    "content": name_prompt
                }
            ],
            model="llama3-70b-8192",
            temperature=0.1,
            max_tokens=100
        )
        
        # Extract names from model response
        names_from_model = chat_completion.choices[0].message.content.strip().split('\n')
        self.participants = sorted([name.strip() for name in names_from_model if name.strip()])
        
        # Process all messages
        current_message = None
        current_text = []
        
        # Split content into lines but keep original formatting
        lines = content.splitlines()
        
        for i, line in enumerate(lines):
            # Skip empty lines or lines that appear to be comments/headers
            if not line or line.startswith("'") or line.startswith('"'):
                continue
                
            # Try to match timestamp pattern
            match = re.match(pattern, line)
            
            if match:
                # If we have a previous message, add it to our list
                if current_message is not None:
                    # Remove emojis from the message
                    clean_message = self.remove_emojis('\n'.join(current_text))
                    current_message['message'] = clean_message
                    messages.append(current_message)
                    current_text = []
                
                # Start a new message
                timestamp, sender, message = match.groups()
                try:
                    # Remove any Unicode spaces before parsing
                    clean_timestamp = timestamp.replace('\u202f', ' ').strip()
                    dt = datetime.strptime(clean_timestamp, '%d/%m/%Y, %I:%M %p')
                    if not "Messages and calls are end-to-end encrypted" in message:
                        current_message = {
                            'timestamp': dt,
                            'sender': sender.strip(),
                            'message': None
                        }
                        # Remove emojis from initial message
                        current_text = [self.remove_emojis(message.strip())]
                    else:
                        current_message = None
                        current_text = []
                except ValueError as e:
                    print(f"Error parsing timestamp: {e}")
                    current_message = None
                    current_text = []
            else:
                # If no timestamp match and we have a current message, 
                # this line is a continuation of the current message
                if current_message is not None and line:
                    # Remove emojis from continuation lines
                    current_text.append(self.remove_emojis(line.strip()))
            
        # Don't forget to add the last message
        if current_message is not None and current_text:
            # Remove emojis from the final message
            clean_message = self.remove_emojis('\n'.join(current_text))
            current_message['message'] = clean_message
            messages.append(current_message)
        
        # Sort messages by timestamp
        messages.sort(key=lambda x: x['timestamp'])
        return messages

    def chunk_messages(self, messages):
        """Split messages into chunks while maintaining token limits and formatting"""
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        # Get participant mapping (A, B) to actual names
        name_mapping = {
            'A':self.participants[0],
            'B':self.participants[1] if len(self.participants) > 1 else None
        }
        reverse_mapping = {v: k for k, v in name_mapping.items()}
        
        # Add name mapping to be included in each chunk
        name_info = f"A is {name_mapping['A']}\nB is {name_mapping['B']}\n\n"
        base_prompt_tokens = self.estimate_tokens(name_info) + 500  # Buffer for prompt template
        
        for i in range(len(messages)):
            current_msg = messages[i]
            sender_letter = reverse_mapping.get(current_msg['sender'])
            if not sender_letter:
                continue
                
            # Format current message
            msg_line = f"{sender_letter}: {current_msg['message']}"
            
            # Check for time delay with next message
            if i < len(messages) - 1:
                next_msg = messages[i + 1]
                time_diff = (next_msg['timestamp'] - current_msg['timestamp']).total_seconds() / 60
                delay_text = self.format_delay(time_diff)
                if delay_text:
                    msg_line += f"\n{delay_text}\n"
            
            # Estimate tokens for this message
            msg_tokens = self.estimate_tokens(msg_line)
            
            # Check if adding this message would exceed token limit
            if current_tokens + msg_tokens + base_prompt_tokens >= 6000:
                if current_chunk:  # Only add if chunk has content
                    chunks.append(name_info + '\n'.join(current_chunk))
                current_chunk = [msg_line]
                current_tokens = msg_tokens + self.estimate_tokens(name_info)
            else:
                current_chunk.append(msg_line)
                current_tokens += msg_tokens
        
        # Add the last chunk if it has content
        if current_chunk:
            chunks.append(name_info + '\n'.join(current_chunk))
        
        return chunks
    
    def estimate_tokens(self, text):
        """
        Estimate number of tokens in text based on word count.
        On average, 100 words is approximately 75 tokens.
        Special characters and numbers may affect the actual token count.
        """
        # Split on whitespace and filter out empty strings
        words = [word for word in text.split() if word.strip()]
        
        # Estimate tokens - typically 75 tokens per 100 words
        estimated_tokens = int(len(words) * 1.5)
        
        # Ensure we always return at least 1 token for non-empty text
        return max(1, estimated_tokens) if text.strip() else 0

    def format_delay(self, minutes):
        """Format time delay into human readable string"""
        if minutes < 60:
            return None  # Return None for delays under threshold
        elif minutes < 1440:  # Less than a day
            hours = int(minutes / 60)
            return f"after {hours} hour{'s' if hours > 1 else ''}"
        else:
            days = int(minutes / 1440)
            return f"{days} day{'s' if days > 1 else ''} later"

    def remove_emojis(self, text):
        """
        Remove emojis from the given text.
        Uses the emoji library to comprehensively remove Unicode emojis.
        """
        # Remove standard emojis
        text_without_emoji = emoji.replace_emoji(text, replace='')
        
        # Remove any remaining emoji-like Unicode characters
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE)
        
        return emoji_pattern.sub(r'', text_without_emoji).strip()

    def log_error(self, error_message, context=None):
        """Log errors to debug file with optional context"""
        try:
            with open('debug/error_log.txt', 'a') as error_file:
                error_file.write(f"\n{'='*50}\n")
                error_file.write(f"Timestamp: {datetime.now()}\n")
                error_file.write(f"Error: {error_message}\n")
                if context:
                    error_file.write(f"Context: {context[:500]}...\n")  # Truncate long contexts
        except Exception as e:
            self.logger.error(f"Failed to write to error log: {e}")

    def truncate_conversation_context(self, conversation_context, max_tokens=5000):
        """
        Truncate conversation context to fit within token limit
        Prioritizes keeping the system message and recent interactions
        """
        total_tokens = 0
        truncated_context = []
        
        # Always keep the system message
        system_message = conversation_context[0]
        truncated_context.append(system_message)
        total_tokens += self.estimate_tokens(system_message['content'])
        
        # Iterate from the end to keep most recent messages
        for message in reversed(conversation_context[1:]):
            msg_tokens = self.estimate_tokens(message['content'])
            
            # If adding this message would exceed max tokens, stop
            if total_tokens + msg_tokens > max_tokens:
                break
            
            # Prepend the message to maintain original order
            truncated_context.insert(1, message)
            total_tokens += msg_tokens
        
        return truncated_context

    def create_continuous_analysis_prompt(self, total_chunks, overall_msgs):
        """Create an initial system prompt for continuous analysis"""
        return f"""You are an expert conversation analyzer focusing on concise, insightful analysis.

PREPROCESSING DETAILS:
- Messages preprocessed: emojis removed, lowercase, no punctuation
- Stopwords eliminated, words stemmed

Analysis Framework for {total_chunks} conversation chunks:
1. Maintain contextual integrity
2. Identify communication transitions
3. Extract core relationship dynamics
4. Highlight communication patterns

Key Analysis Points:
- Semantic nuances
- Emotional undertones
- Communication evolution
- Relationship progression

Chat Participants:
            'A':{self.participants[0]},
            'B':{self.participants[1]}

Provide succinct, meaningful insights. Use the actual names of the participants. After all the chunks, at the end Provide a breif of bondness of the participants over the time period in a Markdown format"""

    def format_response_time_analysis(self, response_analysis):
        """Format response time analysis into readable text"""
        analysis_text = "\nResponse Time Analysis:\n"
        analysis_text += "=====================\n"
        
        for person, stats in response_analysis['stats'].items():
            analysis_text += f"\n{person}:\n"
            analysis_text += f"- Average response time: {self.format_duration(stats['average_response_time'])}\n"
            analysis_text += f"- Fastest response: {self.format_duration(stats['fastest_response'])}\n"
            analysis_text += f"- Slowest response: {self.format_duration(stats['slowest_response'])}\n"
            analysis_text += f"- Total responses: {stats['total_responses']}\n"
            
            # Add response distribution
            dist = stats['response_time_distribution']
            analysis_text += "Response Distribution:\n"
            analysis_text += f"  • Quick responses (< 5 mins): {dist['quick_responses']}\n"
            analysis_text += f"  • Medium responses (5-60 mins): {dist['medium_responses']}\n"
            analysis_text += f"  • Slow responses (> 1 hour): {dist['slow_responses']}\n"

        print(analysis_text)
        analysis_text += "\nDetailed Response Patterns:\n"
        for person, responses in response_analysis['detailed_responses'].items():
            analysis_text += f"\n{person}'s response patterns:\n"
            for resp in responses:
                time_taken = self.format_duration(resp['response_time_minutes'])
                analysis_text += f"- Responded in {time_taken}\n"
                analysis_text += f"  From: {resp['previous_timestamp'].strftime('%Y-%m-%d %H:%M')}\n"
                analysis_text += f"  To: {resp['timestamp'].strftime('%Y-%m-%d %H:%M')}\n"
                analysis_text += f"  To message: '{resp['to_message']}'\n"
                analysis_text += f"  Response: '{resp['response']}'\n"

        return analysis_text

    def format_duration(self, minutes):
        """Convert minutes to a human-readable duration"""
        if minutes < 1:
            return "less than a minute"
        
        days = int(minutes // (24 * 60))
        remaining_minutes = minutes % (24 * 60)
        hours = int(remaining_minutes // 60)
        minutes = int(remaining_minutes % 60)
        
        parts = []
        if days > 0:
            parts.append(f"{days} {'day' if days == 1 else 'days'}")
        if hours > 0:
            parts.append(f"{hours} {'hour' if hours == 1 else 'hours'}")
        if minutes > 0 and days == 0:  # Only show minutes if less than a day
            parts.append(f"{minutes} {'minute' if minutes == 1 else 'minutes'}")
        
        return " ".join(parts)
    
    def analyze_response_patterns(self, messages):
        """Analyze response time patterns between messages"""
        response_times = defaultdict(list)
        conversation_pairs = []
        
        for i in range(1, len(messages)):
            current_msg = messages[i]
            previous_msg = messages[i-1]
            
            # Only analyze if messages are from different senders
            if current_msg['sender'] != previous_msg['sender']:
                time_diff = current_msg['timestamp'] - previous_msg['timestamp']
                minutes_diff = time_diff.total_seconds() / 60
                
                # Skip if the time difference is unreasonably large (e.g., > 30 days)
                if minutes_diff > 43200:  # 30 days
                    continue
                
                # Store response time data
                responder = current_msg['sender']
                response_times[responder].append({
                    'response_time_minutes': minutes_diff,
                    'to_message': previous_msg['message'],
                    'response': current_msg['message'],
                    'timestamp': current_msg['timestamp'],
                    'previous_timestamp': previous_msg['timestamp']
                })
                
                conversation_pairs.append({
                    'initiator': previous_msg['sender'],
                    'responder': current_msg['sender'],
                    'response_time_minutes': minutes_diff,
                    'time_of_day': current_msg['timestamp'].strftime('%H:%M'),
                    'date': current_msg['timestamp'].strftime('%Y-%m-%d')
                })

        # Calculate statistics
        response_stats = {}
        for person, times in response_times.items():
            if times:
                response_times_list = [t['response_time_minutes'] for t in times]
                response_stats[person] = {
                    'average_response_time': sum(response_times_list) / len(response_times_list),
                    'fastest_response': min(response_times_list),
                    'slowest_response': max(response_times_list),
                    'total_responses': len(times),
                    'response_time_distribution': {
                        'quick_responses': len([t for t in response_times_list if t < 5]),  # within 5 minutes
                        'medium_responses': len([t for t in response_times_list if 5 <= t < 60]),  # within an hour
                        'slow_responses': len([t for t in response_times_list if t >= 60])  # more than an hour
                    }
                }

        return {
            'detailed_responses': response_times,
            'stats': response_stats,
            'conversation_pairs': conversation_pairs
        }
    
    def analyze_with_llm(self, messages):
        """Analyze messages with continuous context and preprocessed text"""
        chunks = self.chunk_messages(messages)
        self.logger.info(f"Total Chunks: {len(chunks)}")

        overall_response_analysis = self.analyze_response_patterns(messages)
        self.format_response_time_analysis(overall_response_analysis)
        # Initial system message with context for the entire analysis
        system_message = {
            "role": "system", 
            "content": self.create_continuous_analysis_prompt(len(chunks), overall_response_analysis)
        }

        # Prepare conversation context
        conversation_context = [system_message]
        
        # Analyze chunks
        for i, chunk in enumerate(chunks):
            self.logger.info(f"Processing Chunk {i+1}/{len(chunks)}")
            
            try:
                # Add chunk as a user message
                user_message = {
                    "role": "user",
                    "content": f"Chunk {i+1}: {chunk}"
                }
                conversation_context.append(user_message)
                
                # Truncate conversation context if it's getting too long
                conversation_context = self.truncate_conversation_context(conversation_context)
                
                # If not the last chunk, add a delay
                if i > 0:
                    time.sleep(self.delay)
                
                # Generate analysis for the chunk
                chat_completion = self.client.chat.completions.create(
                    messages=conversation_context,
                    model="llama3-70b-8192",
                    temperature=0.2,
                    max_tokens=2048
                )
                
                # Add model's response to context
                response = chat_completion.choices[0].message.content
                conversation_context.append({
                    "role": "assistant",
                    "content": response
                })
                
                # Track token usage
                self.input_tokens += self.estimate_tokens(chunk)
                self.output_tokens += self.estimate_tokens(response)
                
            except Exception as e:
                error_msg = f"Error processing chunk {i+1}: {str(e)}"
                self.logger.error(error_msg)
                self.log_error(error_msg, chunk)
                # Continue processing next chunk
                continue
        
        # Final analysis prompt
        final_analysis_prompt = {
            "role": "user",
            "content": "Synthesize insights from all conversation chunks. Provide a concise overview of relationship dynamics and communication patterns."
        }
        conversation_context.append(final_analysis_prompt)
        self.format_response_time_analysis(overall_response_analysis)
        try:
            # Generate final comprehensive analysis
            final_completion = self.client.chat.completions.create(
                messages=conversation_context,
                model="llama3-70b-8192",
                temperature=0.2,
                max_tokens=8192
            )
            
            final_analysis = final_completion.choices[0].message.content
            
            return final_analysis, conversation_context
        
        except Exception as e:
            error_msg = f"Error generating final analysis: {str(e)}"
            self.logger.error(error_msg)
            self.log_error(error_msg)
            return "Unable to complete final analysis due to an error."
        
    def extract_and_analyze(self, zip_path):
        """Main method to extract and analyze chat"""
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                temp_dir = "temp_extracted"
                zip_ref.extractall(temp_dir)
                
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        if file.endswith('.txt'):
                            file_path = os.path.join(root, file)
                            with open(file_path, 'r', encoding='utf-8') as f:
                                self.chat_content = f.read()
                
                # Process messages
                self.processed_messages = self.process_message(self.chat_content)
                
                # Get chunked analysis
                analysis, context = self.analyze_with_llm(self.processed_messages)
                
                # Clean up
                self._cleanup(temp_dir)
                
                return analysis, context

        except Exception as e:
            print(f"Error occurred: {str(e)}")
            return None

    def _cleanup(self, temp_dir):
        """Remove temporary directory and its contents"""
        for root, dirs, files in os.walk(temp_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(temp_dir)

    def preprocess_text(self, text):
        """Preprocess the input text for analysis."""
        # Example preprocessing: convert to lowercase and remove punctuation
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        return text.strip()

    def generate_query_response(self, user_query, context):
        # Append user query to conversation context
        context.append({
            "role": "user",
            "content": user_query
        })
        
        # Generate response with full context
        response = self.client.chat.completions.create(
            messages=context,
            model="llama3-70b-8192",
            temperature=0.1,
            max_tokens=8192
        )
        
        # Add model's response to context
        model_response = response.choices[0].message.content
        context.append({
            "role": "assistant",
            "content": model_response
        })
        
        return model_response
    
# Example usage
if __name__ == "__main__":
    analyzer = EnhancedChunkedChatAnalyzer(chunk_size=75, delay_between_requests=4)
    file_path = "/home/user/path/to/WA_conversation_without_media.zip"
    pickle_file_path = file_path + ".pickle"
    if not os.path.exists(pickle_file_path):
        analysis, context = analyzer.extract_and_analyze(file_path)
        analyzer.save_conversation_context(context, pickle_file_path)
        if analysis:
            print("\nChat Analysis Results:")
            print("=====================")
            print(analysis)
            
            # Additional query prompt
            while True:
                additional_query = input("\nWould you like to ask any additional questions about the chat? (Type 'exit' to quit): ")
                if additional_query.lower() == 'exit':
                    break
                
                print(analyzer.generate_query_response(additional_query, context))
    else:
        context = analyzer.load_conversation_context(pickle_file_path)
        while True:
            additional_query = input("\nWould you like to ask any additional questions about the loaded Context chat? (Type 'exit' to quit): ")
            if additional_query.lower() == 'exit':
                break
            
            print(analyzer.generate_query_response(additional_query, context))
