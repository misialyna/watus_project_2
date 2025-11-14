#!/usr/bin/env python3
"""
Standalone Watus HTML Interface
Niezależny monitor emotikonów dla watus_project
Podłącza się przez ZMQ do istniejących topiców bez modyfikacji watus.py
"""

import zmq
import json
import threading
import time
import re
from datetime import datetime
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit
import os
from dotenv import load_dotenv

load_dotenv()

class StandaloneWatusInterface:
    def __init__(self):
        # Flask app setup
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'standalone-watus-interface'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # ZMQ Setup - słuchamy istniejących topiców z watus.py
        self.context = zmq.Context()
        
        # Subscriber dla istniejących topiców watus.py
        self.watus_subscriber = self.context.socket(zmq.SUB)
        self.watus_subscriber.connect("tcp://127.0.0.1:7780")  # PUB port z watus.py
        
        # Subscribe to existing topics
        self.watus_subscriber.setsockopt_string(zmq.SUBSCRIBE, "dialog.leader")
        self.watus_subscriber.setsockopt_string(zmq.SUBSCRIBE, "unknown_utterance")
        
        # Journal monitoring - śledzimy pliki logów watus
        self.log_files = {
            'watus': 'watus.log',
            'dialog': 'dialog.jsonl',
            'camera': 'camera.jsonl'
        }
        
        self.current_emotion = "standby"
        self.last_watus_state = "unknown"
        self.last_speaker = None
        self.running = False
        
        # Advanced state detection patterns
        self.state_patterns = {
            'listening': [
                r'listening.*for.*wake',
                r'voice.*activity.*detected',
                r'waiting.*for.*speech'
            ],
            'processing': [
                r'processing.*utterance',
                r'transcribing',
                r'speaker.*recognition',
                r'verification'
            ],
            'speaking': [
                r'speaking.*response',
                r'tts.*playing',
                r'audio.*output'
            ],
            'happy': [
                r'leader.*identified',
                r'speaker.*verified',
                r'enrollment.*successful'
            ],
            'error': [
                r'error.*',
                r'failed.*',
                r'exception',
                r'timeout'
            ],
            'sleep': [
                r'sleeping',
                r'inactive',
                r'standalone.*mode'
            ]
        }
        
        self._setup_routes()
        self._setup_socketio_handlers()
    
    def _setup_routes(self):
        """Setup Flask routes"""
        @self.app.route('/')
        def index():
            return render_template('watus_face.html')
        
        @self.app.route('/status')
        def status():
            return jsonify({
                'status': 'running',
                'current_emotion': self.current_emotion,
                'last_watus_state': self.last_watus_state,
                'last_speaker': self.last_speaker,
                'connected_clients': self.socketio.server.manager.get_participants() if hasattr(self.socketio.server, 'manager') else 0
            })
        
        @self.app.route('/health')
        def health():
            return jsonify({
                'health': 'ok',
                'timestamp': datetime.now().isoformat(),
                'zmq_connected': self._check_zmq_connection()
            })
        
        @self.app.route('/watus_status')
        def watus_status():
            """Get detailed watus project status"""
            return jsonify({
                'last_watus_state': self.last_watus_state,
                'last_speaker': self.last_speaker,
                'current_emotion': self.current_emotion,
                'timestamp': datetime.now().isoformat()
            })
    
    def _setup_socketio_handlers(self):
        """Setup WebSocket event handlers"""
        @self.socketio.on('connect')
        def on_connect():
            print(f'✅ Client connected to Standalone Watus Interface')
            emit('emotion_update', {'emotion': self.current_emotion})
            emit('watus_status', {
                'last_state': self.last_watus_state,
                'last_speaker': self.last_speaker
            })
        
        @self.socketio.on('disconnect')
        def on_disconnect():
            print('❌ Client disconnected')
        
        @self.socketio.on('request_emotion')
        def on_request_emotion(data):
            """Manual emotion request from client"""
            emotion = data.get('emotion')
            if emotion in ['standby', 'listening', 'processing', 'speaking', 'happy', 'error', 'sleep']:
                self.set_emotion(emotion, source='manual')
                emit('emotion_update', {'emotion': emotion})
    
    def _check_zmq_connection(self):
        """Check if ZMQ connection to watus is working"""
        try:
            # Try to get one message to verify connection
            poller = zmq.Poller()
            poller.register(self.watus_subscriber, zmq.POLLIN)
            socks = dict(poller.poll(timeout=1))
            return self.watus_subscriber in socks
        except:
            return False
    
    def set_emotion(self, emotion, source='auto'):
        """Set current emotion and broadcast to all clients"""
        if emotion == self.current_emotion:
            return
            
        self.current_emotion = emotion
        self.last_watus_state = emotion
        
        # Broadcast to all connected clients via WebSocket
        self.socketio.emit('emotion_update', {'emotion': emotion})
        self.socketio.emit('watus_status', {
            'last_state': emotion,
            'last_speaker': self.last_speaker,
            'source': source
        })
        
        print(f"🎭 Standalone Watus emotion: {emotion} (source: {source})")
    
    def analyze_dialog_message(self, topic, message):
        """Analyze dialog messages from watus.py to determine state"""
        try:
            if topic == "dialog.leader":
                # Parse JSON message from watus.py
                dialog_data = json.loads(message)
                
                speaker = dialog_data.get('speaker', 'unknown')
                text = dialog_data.get('text', '')
                timestamp = dialog_data.get('timestamp', '')
                
                self.last_speaker = speaker
                
                if speaker == 'leader':
                    self.set_emotion('happy', source='dialog_leader')
                else:
                    self.set_emotion('standby', source='dialog_unknown')
                    
                print(f"📝 Dialog: {speaker} said: {text[:50]}...")
                
            elif topic == "unknown_utterance":
                self.set_emotion('standby', source='dialog_unknown')
                print(f"❓ Unknown utterance: {message[:50]}...")
                
        except json.JSONDecodeError:
            print(f"⚠️  Could not parse dialog message: {message[:100]}")
        except Exception as e:
            print(f"❌ Error analyzing dialog message: {e}")
    
    def monitor_log_files(self):
        """Monitor watus log files for state changes"""
        print("📄 Monitoring watus log files...")
        
        log_positions = {}
        
        while self.running:
            try:
                for log_name, log_path in self.log_files.items():
                    if os.path.exists(log_path):
                        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                            # Seek to last position
                            if log_name in log_positions:
                                f.seek(log_positions[log_name])
                            else:
                                f.seek(0, 2)  # Start from end
                            
                            # Read new lines
                            new_lines = f.readlines()
                            
                            if new_lines:
                                # Update position
                                log_positions[log_name] = f.tell()
                                
                                # Analyze new lines
                                for line in new_lines[-3:]:  # Last 3 lines
                                    self._analyze_log_line(line.strip())
                    else:
                        # Log file doesn't exist yet
                        pass
                
                time.sleep(2)  # Check every 2 seconds
                
            except Exception as e:
                print(f"❌ Log monitoring error: {e}")
                time.sleep(5)
    
    def _analyze_log_line(self, line):
        """Analyze a single log line for state patterns"""
        if not line:
            return
            
        line_lower = line.lower()
        
        # Check against state patterns
        for state, patterns in self.state_patterns.items():
            for pattern in patterns:
                if re.search(pattern, line_lower, re.IGNORECASE):
                    # Only change emotion if it's different
                    if state != self.current_emotion:
                        self.set_emotion(state, source='log_analysis')
                        print(f"📝 Log analysis: {state} detected from: {line[:80]}...")
                    return
        
        # Additional specific checks
        if 'led.listening' in line_lower:
            self.set_emotion('listening', source='led_state')
        elif 'processing' in line_lower and 'speaking' in line_lower:
            self.set_emotion('processing', source='led_state')
        elif 'cue_speak' in line_lower or 'tts' in line_lower:
            self.set_emotion('speaking', source='tts_state')
    
    def monitor_zmq_messages(self):
        """Monitor ZMQ messages from watus.py"""
        print("📡 Monitoring ZMQ messages from watus.py...")
        
        poller = zmq.Poller()
        poller.register(self.watus_subscriber, zmq.POLLIN)
        
        while self.running:
            try:
                # Poll for messages with timeout
                socks = dict(poller.poll(timeout=1000))
                
                if self.watus_subscriber in socks:
                    message = self.watus_subscriber.recv_string()
                    
                    # Extract topic and message
                    if ' ' in message:
                        topic, data = message.split(' ', 1)
                        self.analyze_dialog_message(topic, data)
                    else:
                        print(f"⚠️  Invalid ZMQ message format: {message[:100]}")
                
            except zmq.Again:
                # Timeout - continue
                pass
            except Exception as e:
                print(f"❌ ZMQ monitoring error: {e}")
                time.sleep(1)
    
    def _demo_mode(self):
        """Demo mode - cycles through emotions automatically"""
        emotions = ['standby', 'listening', 'processing', 'speaking', 'happy', 'error', 'sleep']
        emotion_index = 0
        
        while self.running:
            if self.current_emotion == 'standby':  # Only cycle when in standby
                emotion = emotions[emotion_index % len(emotions)]
                self.set_emotion(emotion, source='demo')
                emotion_index += 1
            
            time.sleep(8)  # Change every 8 seconds
    
    def run_interface(self, host='127.0.0.1', port=5000):
        """Start the standalone HTML interface"""
        self.running = True
        
        # Start monitoring threads
        zmq_thread = threading.Thread(target=self.monitor_zmq_messages, daemon=True)
        zmq_thread.start()
        
        log_thread = threading.Thread(target=self.monitor_log_files, daemon=True)
        log_thread.start()
        
        # Demo mode thread
        demo_thread = threading.Thread(target=self._demo_mode, daemon=True)
        demo_thread.start()
        
        print(f"🌐 Starting Standalone Watus HTML Interface on http://{host}:{port}")
        print("🔗 Auto-connected to watus_project via ZMQ (no modifications needed)")
        print("📄 Monitoring log files: watus.log, dialog.jsonl, camera.jsonl")
        print("📡 Subscribed to ZMQ topics: dialog.leader, unknown_utterance")
        print("🔄 Demo mode active - cycling emotions every 8 seconds")
        
        # Run Flask-SocketIO server
        self.socketio.run(self.app, host=host, port=port, debug=False)
    
    def stop(self):
        """Stop the interface"""
        self.running = False
        self.context.term()

if __name__ == "__main__":
    # Configuration from environment
    host = os.getenv('STANDALONE_INTERFACE_HOST', '127.0.0.1')
    port = int(os.getenv('STANDALONE_INTERFACE_PORT', 5001))  # Different port by default
    
    print("🤖 Standalone Watus HTML Interface")
    print("=" * 50)
    print("Features:")
    print("✅ No modifications to watus.py required")
    print("✅ Auto-connects via existing ZMQ topics")
    print("✅ Monitors log files for state changes")
    print("✅ Real-time emoticon updates")
    print("✅ Multi-client support")
    print("=" * 50)
    
    interface = StandaloneWatusInterface()
    
    try:
        interface.run_interface(host, port)
    except KeyboardInterrupt:
        print("\n🛑 Stopping Standalone Watus HTML Interface...")
        interface.stop()