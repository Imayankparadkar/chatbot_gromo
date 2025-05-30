from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import json
from datetime import datetime
import requests
import logging
import uuid
import os
from werkzeug.exceptions import BadRequest

app = Flask(__name__)

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Enhanced CORS configuration
# Update your CORS configuration
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:3000", "http://127.0.0.1:5000", "*"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
    }
})

# Groq API configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Enhanced conversation storage
conversations = {}
MAX_CONVERSATIONS = 100
MAX_MESSAGES_PER_CONVERSATION = 50

def clean_old_conversations():
    """Remove old conversations to prevent memory overflow"""
    if len(conversations) > MAX_CONVERSATIONS:
        sorted_sessions = sorted(conversations.keys())
        for session_id in sorted_sessions[:10]:
            del conversations[session_id]
        logger.info(f"Cleaned old conversations. Current count: {len(conversations)}")

def generate_enhanced_system_prompt(agent_name=None, client_name=None, client_age=None, 
                                   client_income=None, client_goal=None, language="English"):
    """Generate an enhanced system prompt for comprehensive responses"""
    base_info = ""
    if agent_name and client_name:
        base_info = f"""
### 👥 Session Info:
- **Gromo Agent**: {agent_name}
- **Client**: {client_name}
- **Age**: {client_age or 'Not specified'}
- **Monthly Income**: ₹{client_income or 'Not specified'}
- **Goal**: {client_goal or 'General financial guidance'}
- **Language**: {language}
"""
    
    return f"""
You are **Gromo Coach**, an advanced AI financial advisor for Gromo - India's leading financial services platform. You provide comprehensive, detailed responses with practical examples.

{base_info}
---
### 🎯 RESPONSE STRUCTURE (MANDATORY):

For EVERY user question, structure your response exactly like this:

**📝 QUICK SUMMARY:**
[1-2 sentence summary of the answer]

**📚 DETAILED EXPLANATION:**
[Comprehensive explanation broken into clear sections with headings]

**💡 PRACTICAL EXAMPLES:**
[Provide 2-3 real-world examples, preferably Indian context]

**🔧 ACTIONABLE STEPS:**
[Give specific, actionable advice the user can implement]

**❓ RELATED QUESTIONS TO EXPLORE:**
[Suggest 3-4 relevant follow-up questions they might want to ask]

### 🎯 Your Enhanced Capabilities:

**Financial Expertise Areas:**
- Insurance, Investments, Banking, Tax Planning, Real Estate, etc.

**Communication Style:**
- Be conversational but professional
- Use Indian examples and context
- Include relevant numbers and calculations
- Provide actionable advice

IMPORTANT: Every response must follow the 5-section structure. Never skip any section.
"""

def validate_groq_api_key():
    """Validate if Groq API key is working"""
    try:
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        test_payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 10
        }
        
        logger.debug("Testing Groq API connection...")
        response = requests.post(GROQ_API_URL, headers=headers, json=test_payload, timeout=10)
        logger.debug(f"Groq API test response: {response.status_code}")
        
        if response.status_code == 200:
            return True
        else:
            logger.error(f"API validation failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"Groq API validation error: {str(e)}")
        return False

def generate_groq_response(messages, max_tokens=2000, temperature=0.7):
    """Generate AI response using Groq API with enhanced error handling"""
    try:
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 1,
            "stream": False
        }
        
        logger.debug(f"Sending request to Groq API with {len(messages)} messages")
        logger.debug(f"Payload: {json.dumps(payload, indent=2)}")
        
        response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=60)
        
        logger.debug(f"Groq API response status: {response.status_code}")
        logger.debug(f"Response headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            logger.debug(f"Full API response: {json.dumps(result, indent=2)}")
            
            if "choices" in result and len(result["choices"]) > 0:
                ai_response = result["choices"][0]["message"]["content"]
                logger.info(f"Successfully generated response from Groq API (length: {len(ai_response)})")
                return ai_response
            else:
                logger.error("No choices in API response")
                return generate_fallback_response("API returned no response choices")
        else:
            error_text = response.text
            logger.error(f"Groq API Error: {response.status_code} - {error_text}")
            
            # Handle specific error cases
            if response.status_code == 401:
                return generate_fallback_response("API authentication failed - please check API key")
            elif response.status_code == 429:
                return generate_fallback_response("API rate limit exceeded - please try again later")
            elif response.status_code >= 500:
                return generate_fallback_response("API server error - please try again")
            else:
                return generate_fallback_response(f"API Error {response.status_code}")
            
    except requests.exceptions.Timeout:
        logger.error("Groq API timeout")
        return generate_fallback_response("The AI is taking longer than usual to respond. Please try again.")
    except requests.exceptions.ConnectionError:
        logger.error("Connection error to Groq API")
        return generate_fallback_response("Unable to connect to AI service. Please check your internet connection.")
    except requests.exceptions.RequestException as e:
        logger.error(f"Request Error: {str(e)}")
        return generate_fallback_response(f"Network error: {str(e)}")
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {str(e)}")
        return generate_fallback_response("Invalid response format from AI service")
    except Exception as e:
        logger.error(f"Unexpected Error in generate_groq_response: {str(e)}")
        return generate_fallback_response(f"Unexpected error: {str(e)}")

def generate_fallback_response(custom_message=None):
    """Enhanced fallback response with proper structure"""
    base_message = custom_message or "I'm experiencing technical difficulties but I'm here to help."
    
    return f"""**📝 QUICK SUMMARY:**
{base_message}

**📚 DETAILED EXPLANATION:**
I'm Gromo Coach, designed to provide comprehensive financial guidance including insurance, investments, tax planning, and banking solutions. Currently experiencing some technical issues, but I'll do my best to help you.

**💡 PRACTICAL EXAMPLES:**
- Life insurance planning for young professionals (₹50 lakh coverage for ₹500/month)
- SIP investments for wealth building (₹5000/month can grow to ₹50 lakh in 15 years)
- Tax-saving strategies under Section 80C (save up to ₹46,800 in taxes)

**🔧 ACTIONABLE STEPS:**
1. Please try your question again in a moment
2. Be specific about your financial situation (age, income, goals)
3. Mention your investment timeline for better advice
4. Consider speaking with a financial advisor for complex queries

**❓ RELATED QUESTIONS TO EXPLORE:**
- What's the best investment strategy for someone my age?
- How much life insurance coverage do I actually need?
- Which tax-saving options give the best returns?
- How should I create and maintain an emergency fund?"""

@app.before_request
def before_request():
    """Handle preflight requests and log all requests"""
    logger.debug(f"Incoming request: {request.method} {request.path}")
    
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        return response

@app.route('/')
def index():
    """Serve main page with enhanced testing"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Gromo Coach API - Fixed Version</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .status { padding: 15px; border-radius: 5px; margin: 15px 0; }
            .success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
            .feature { background: #e7f3ff; padding: 10px; margin: 5px 0; border-left: 4px solid #007bff; }
            .endpoint { background: #f8f9fa; padding: 15px; margin: 10px 0; border-left: 4px solid #007bff; }
            h1 { color: #333; }
            .test-section { background: #fff3cd; padding: 20px; margin: 20px 0; border-radius: 5px; }
            button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; margin: 5px; }
            button:hover { background: #0056b3; }
            #testResult { margin-top: 10px; padding: 10px; border-radius: 5px; max-height: 400px; overflow-y: auto; }
            .chat-test { background: #f8f9fa; padding: 20px; margin: 20px 0; border-radius: 5px; }
            textarea { width: 100%; height: 80px; margin: 10px 0; padding: 10px; border-radius: 5px; border: 1px solid #ddd; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 Gromo Coach API - Fixed Version</h1>
            <div class="status success">
                ✅ Server is running with enhanced error handling and debugging!
            </div>
            
            <div class="test-section">
                <h3>🧪 Quick Test Section</h3>
                <button onclick="testHealth()">Test Health</button>
                <button onclick="testGroq()">Test Groq API</button>
                <button onclick="testBasicChat()">Test Basic Chat</button>
                <div id="testResult"></div>
            </div>
            
            <div class="chat-test">
                <h3>💬 Full Chat Test</h3>
                <textarea id="chatMessage" placeholder="Enter your message here (e.g., 'Tell me about SIP investments')">Tell me about SIP investments for a 25-year-old earning ₹50,000 per month</textarea><br>
                <button onclick="testFullChat()">Send Chat Message</button>
                <div id="chatResult"></div>
            </div>
            
            <h3>🎯 Response Structure:</h3>
            <div class="feature">📝 <strong>Quick Summary</strong></div>
            <div class="feature">📚 <strong>Detailed Explanation</strong></div>
            <div class="feature">💡 <strong>Practical Examples</strong></div>
            <div class="feature">🔧 <strong>Actionable Steps</strong></div>
            <div class="feature">❓ <strong>Related Questions</strong></div>
            
            <h3>Available Endpoints:</h3>
            <div class="endpoint"><strong>POST /chat</strong> - Main chat endpoint</div>
            <div class="endpoint"><strong>GET /health</strong> - Health check</div>
            <div class="endpoint"><strong>GET /test_groq</strong> - Test Groq API</div>
            <div class="endpoint"><strong>GET /debug</strong> - Debug information</div>
        </div>
        
        <script>
            async function testHealth() {
                const resultDiv = document.getElementById('testResult');
                resultDiv.innerHTML = 'Testing health endpoint...';
                
                try {
                    const response = await fetch('/health');
                    const data = await response.json();
                    resultDiv.innerHTML = `<div style="background: ${response.ok ? '#d4edda' : '#f8d7da'}; padding: 10px; border-radius: 5px;">
                        <strong>Status:</strong> ${data.status}<br>
                        <strong>Groq API:</strong> ${data.groq_api_status}<br>
                        <strong>Active Conversations:</strong> ${data.active_conversations}
                    </div>`;
                } catch (error) {
                    resultDiv.innerHTML = `<div style="background: #f8d7da; padding: 10px; border-radius: 5px;">
                        <strong>Error:</strong> ${error.message}
                    </div>`;
                }
            }
            
            async function testGroq() {
                const resultDiv = document.getElementById('testResult');
                resultDiv.innerHTML = 'Testing Groq API...';
                
                try {
                    const response = await fetch('/test_groq');
                    const data = await response.json();
                    resultDiv.innerHTML = `<div style="background: ${response.ok ? '#d4edda' : '#f8d7da'}; padding: 10px; border-radius: 5px;">
                        <strong>Status:</strong> ${data.status}<br>
                        <strong>API Valid:</strong> ${data.api_key_valid}<br>
                        <strong>Message:</strong> ${data.message}<br>
                        <strong>Test Response:</strong> ${data.test_response ? data.test_response.substring(0, 200) + '...' : 'None'}
                    </div>`;
                } catch (error) {
                    resultDiv.innerHTML = `<div style="background: #f8d7da; padding: 10px; border-radius: 5px;">
                        <strong>Error:</strong> ${error.message}
                    </div>`;
                }
            }
            
            async function testBasicChat() {
                const resultDiv = document.getElementById('testResult');
                resultDiv.innerHTML = 'Testing basic chat...';
                
                try {
                    const response = await fetch('/chat', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            message: 'Hello, can you help me?',
                            session_id: 'test-session-' + Date.now()
                        })
                    });
                    
                    const data = await response.json();
                    resultDiv.innerHTML = `<div style="background: ${response.ok ? '#d4edda' : '#f8d7da'}; padding: 10px; border-radius: 5px;">
                        <strong>Status:</strong> ${response.status}<br>
                        <strong>Success:</strong> ${data.success}<br>
                        <strong>Response Preview:</strong> ${data.response ? data.response.substring(0, 300) + '...' : 'No response'}<br>
                        <strong>Error:</strong> ${data.error || 'None'}
                    </div>`;
                } catch (error) {
                    resultDiv.innerHTML = `<div style="background: #f8d7da; padding: 10px; border-radius: 5px;">
                        <strong>Error:</strong> ${error.message}
                    </div>`;
                }
            }
            
            async function testFullChat() {
                const message = document.getElementById('chatMessage').value;
                const resultDiv = document.getElementById('chatResult');
                resultDiv.innerHTML = 'Sending message...';
                
                try {
                    const response = await fetch('/chat', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            message: message,
                            session_id: 'full-test-' + Date.now(),
                            client_name: 'Test User',
                            client_age: '25',
                            client_income: '50000'
                        })
                    });
                    
                    const data = await response.json();
                    if (data.success) {
                        resultDiv.innerHTML = `<div style="background: #d4edda; padding: 15px; border-radius: 5px; white-space: pre-wrap;">${data.response}</div>`;
                    } else {
                        resultDiv.innerHTML = `<div style="background: #f8d7da; padding: 15px; border-radius: 5px;">
                            <strong>Error:</strong> ${data.error}<br>
                            <strong>Details:</strong> ${data.details || 'None'}
                        </div>`;
                    }
                } catch (error) {
                    resultDiv.innerHTML = `<div style="background: #f8d7da; padding: 15px; border-radius: 5px;">
                        <strong>Network Error:</strong> ${error.message}
                    </div>`;
                }
            }
        </script>
    </body>
    </html>
    """

@app.route('/chat', methods=['POST'])
def chat():
    """Enhanced chat endpoint with detailed debugging"""
    try:
        logger.debug("=== CHAT ENDPOINT CALLED ===")
        
        # Clean old conversations periodically
        clean_old_conversations()
        
        # Get and validate JSON data
        if not request.is_json:
            logger.error("Request is not JSON")
            return jsonify({
                'error': 'Request must be JSON',
                'success': False
            }), 400
        
        data = request.get_json()
        logger.debug(f"Received data: {data}")
        
        if not data:
            logger.error("No JSON data received")
            return jsonify({
                'error': 'No JSON data received',
                'success': False
            }), 400
        
        # Extract and validate data
        session_id = data.get('session_id', str(uuid.uuid4()))
        user_message = data.get('message', '').strip()
        agent_name = data.get('agent_name', '')
        client_name = data.get('client_name', '')
        client_age = data.get('client_age', '')
        client_income = data.get('client_income', '')
        client_goal = data.get('client_goal', '')
        language = data.get('language', 'English')
        reset_conversation = data.get('reset', False)
        
        logger.info(f"Processing message for session {session_id}: '{user_message}'")
        
        if not user_message:
            logger.error("Empty user message")
            return jsonify({
                'error': 'Message is required and cannot be empty',
                'success': False
            }), 400
        
        # Initialize or reset conversation
        if session_id not in conversations or reset_conversation:
            system_prompt = generate_enhanced_system_prompt(
                agent_name, client_name, client_age, client_income, client_goal, language
            )
            conversations[session_id] = [
                {"role": "system", "content": system_prompt}
            ]
            logger.info(f"Initialized conversation for session {session_id}")
        
        # Add user message to conversation
        conversations[session_id].append({"role": "user", "content": user_message})
        logger.debug(f"Conversation length: {len(conversations[session_id])}")
        
        # Generate AI response
        logger.info("Generating AI response...")
        ai_response = generate_groq_response(conversations[session_id], max_tokens=2000)
        
        if not ai_response:
            logger.error("Empty AI response generated")
            ai_response = generate_fallback_response("Failed to generate response from AI service")
        
        # Add AI response to conversation history
        conversations[session_id].append({"role": "assistant", "content": ai_response})
        
        # Keep conversation history manageable
        if len(conversations[session_id]) > MAX_MESSAGES_PER_CONVERSATION:
            conversations[session_id] = [conversations[session_id][0]] + conversations[session_id][-40:]
            logger.info(f"Trimmed conversation history for session {session_id}")
        
        response_data = {
            'success': True,
            'response': ai_response,
            'session_id': session_id,
            'message_count': len(conversations[session_id]) - 1,
            'timestamp': datetime.now().isoformat(),
            'response_type': 'comprehensive_structured'
        }
        
        logger.info(f"Successfully generated response (length: {len(ai_response)})")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Chat endpoint error: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'An internal error occurred',
            'success': False,
            'details': str(e)
        }), 500

@app.route('/debug', methods=['GET'])
def debug_info():
    """Debug endpoint to check server status"""
    groq_status = validate_groq_api_key()
    return jsonify({
        'server_status': 'running',
        'groq_api_key_configured': bool(GROQ_API_KEY),
        'groq_api_key_length': len(GROQ_API_KEY) if GROQ_API_KEY else 0,
        'groq_api_working': groq_status,
        'active_conversations': len(conversations),
        'conversation_ids': list(conversations.keys())[:5],
        'environment_variables': {
            'GROQ_API_KEY_SET': 'GROQ_API_KEY' in os.environ
        },
        'timestamp': datetime.now().isoformat()
    })

# Add this route to your Flask app (after your existing /chat route)

@app.route('/api/chat', methods=['POST'])
def api_chat():
    """API endpoint that matches the frontend expectation"""
    try:
        logger.debug("=== API CHAT ENDPOINT CALLED ===")
        
        # Get JSON data
        if not request.is_json:
            return jsonify({
                'error': 'Request must be JSON',
                'success': False
            }), 400
        
        data = request.get_json()
        logger.debug(f"API endpoint received data: {data}")
        
        # Extract data with frontend naming conventions
        session_id = data.get('sessionId') or data.get('session_id', str(uuid.uuid4()))
        user_message = data.get('message', '').strip()
        language = data.get('language', 'English')
        
        if not user_message:
            return jsonify({
                'error': 'Message is required',
                'success': False
            }), 400
        
        # Use the same logic as your existing chat endpoint
        if session_id not in conversations:
            system_prompt = generate_enhanced_system_prompt(language=language)
            conversations[session_id] = [
                {"role": "system", "content": system_prompt}
            ]
        
        # Add user message
        conversations[session_id].append({"role": "user", "content": user_message})
        
        # Generate AI response
        ai_response = generate_groq_response(conversations[session_id], max_tokens=2000)
        
        if not ai_response:
            ai_response = generate_fallback_response("Failed to generate response")
        
        # Add AI response to conversation
        conversations[session_id].append({"role": "assistant", "content": ai_response})
        
        # Return response in the format your frontend expects
        return jsonify({
            'success': True,
            'response': ai_response,  # Frontend looks for 'response'
            'message': ai_response,   # Alternative field name
            'session_id': session_id,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"API chat endpoint error: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'An internal error occurred',
            'success': False,
            'details': str(e)
        }), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    groq_status = validate_groq_api_key()
    return jsonify({
        'status': 'healthy',
        'message': 'Fixed Gromo Coach API is running!',
        'groq_api_status': 'connected' if groq_status else 'disconnected',
        'active_conversations': len(conversations),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/test_groq', methods=['GET'])
def test_groq():
    """Test Groq API connectivity with enhanced response"""
    try:
        test_messages = [
            {"role": "system", "content": "You are Gromo Coach. Respond with the 5-section structure."},
            {"role": "user", "content": "Test if you're working properly with a simple hello."}
        ]
        test_response = generate_groq_response(test_messages, max_tokens=500)
        
        return jsonify({
            'status': 'success',
            'message': 'Groq API is working!',
            'test_response': test_response,
            'api_key_valid': True,
            'response_length': len(test_response) if test_response else 0
        })
    except Exception as e:
        logger.error(f"Groq API test failed: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Groq API test failed: {str(e)}',
            'api_key_valid': False
        }), 500

# Additional utility endpoints
@app.route('/conversation_history/<session_id>', methods=['GET'])
def get_conversation_history(session_id):
    """Get conversation history for a session"""
    try:
        if session_id in conversations:
            history = [msg for msg in conversations[session_id] if msg['role'] != 'system']
            return jsonify({
                'success': True,
                'history': history,
                'message_count': len(history),
                'session_id': session_id
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Session not found',
                'history': []
            }), 404
    except Exception as e:
        logger.error(f"Error retrieving conversation history: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Failed to retrieve conversation history'
        }), 500

@app.route('/clear_conversation/<session_id>', methods=['POST'])
def clear_conversation(session_id):
    """Clear conversation history for a session"""
    try:
        if session_id in conversations:
            del conversations[session_id]
            logger.info(f"Cleared conversation for session {session_id}")
        return jsonify({
            'success': True,
            'message': 'Conversation cleared successfully'
        })
    except Exception as e:
        logger.error(f"Error clearing conversation: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Failed to clear conversation'
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint not found',
        'message': f'The requested endpoint does not exist'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred'
    }), 500

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 Starting FIXED Gromo Coach API")
    print("=" * 60)
    
    logger.info("Starting server...")
    logger.info(f"Groq API Key configured: {'Yes' if GROQ_API_KEY else 'No'}")
    
    # Test API on startup
    if validate_groq_api_key():
        logger.info("✅ Groq API connection successful!")
    else:
        logger.warning("⚠️ Groq API connection failed - check your API key")
    
    print("\n🧪 TROUBLESHOOTING STEPS:")
    print("1. Visit http://localhost:5000 for testing interface")
    print("2. Click 'Test Groq API' to verify API connection")
    print("3. Try 'Test Basic Chat' for simple message test")
    print("4. Use the full chat test with your custom message")
    print("5. Check console logs for detailed error information")
    print("=" * 60)
    
    app.run(
        debug=True, 
        host='0.0.0.0', 
        port=5000, 
        threaded=True
    )