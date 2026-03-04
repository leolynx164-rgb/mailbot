#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backend Flask pour Email AI Auto-Responder
Gère IMAP, SMTP et les appels API IA
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import imaplib
import smtplib
import email
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.header import decode_header
import threading
import time
import requests
from datetime import datetime
import json

app = Flask(__name__, static_folder='.')
CORS(app)

# État global de la surveillance
monitoring_state = {
    'running': False,
    'thread': None,
    'config': {},
    'stats': {
        'emails_scanned': 0,
        'emails_responded': 0,
        'errors': 0,
        'start_time': None
    },
    'logs': [],
    'processed_uids': set()
}

def log_message(message, level="INFO"):
    """Ajouter un message au log"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = {
        'timestamp': timestamp,
        'level': level,
        'message': message
    }
    monitoring_state['logs'].append(log_entry)
    # Garder seulement les 200 derniers logs
    if len(monitoring_state['logs']) > 200:
        monitoring_state['logs'] = monitoring_state['logs'][-200:]
    print(f"[{timestamp}] [{level}] {message}")

def decode_header_value(header_value):
    """Décoder un en-tête d'email"""
    if not header_value:
        return ""
    decoded_parts = decode_header(header_value)
    decoded_string = ""
    for part, encoding in decoded_parts:
        if isinstance(part, bytes):
            decoded_string += part.decode(encoding or 'utf-8', errors='ignore')
        else:
            decoded_string += part
    return decoded_string

def get_email_body(msg):
    """Extraire le corps du message email"""
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition"))
            
            if "attachment" not in content_disposition:
                if content_type == "text/plain":
                    try:
                        payload = part.get_payload(decode=True)
                        if payload:
                            body += payload.decode('utf-8', errors='ignore')
                    except:
                        pass
                elif content_type == "text/html" and not body:
                    try:
                        payload = part.get_payload(decode=True)
                        if payload:
                            body += payload.decode('utf-8', errors='ignore')
                    except:
                        pass
    else:
        try:
            payload = msg.get_payload(decode=True)
            if payload:
                body = payload.decode('utf-8', errors='ignore')
        except:
            pass
    
    return body.strip()

def generate_ai_response(provider, api_key, model, ollama_url, context, sender, subject, body):
    """Générer une réponse avec l'IA"""
    prompt = f"""Tu es un assistant email professionnel. Voici le contexte personnalisé pour tes réponses:

{context}

Un email a été reçu avec les informations suivantes:
- Expéditeur: {sender}
- Sujet: {subject}
- Corps du message:
{body[:1000]}

En tenant compte du contexte fourni, génère une réponse professionnelle, courtoise et pertinente à cet email. 
La réponse doit être en français, bien structurée et adaptée au contexte.
Réponds uniquement avec le contenu de l'email, sans formule de signature (elle sera ajoutée automatiquement).
"""
    
    try:
        if provider == "gemini":
            return generate_gemini(api_key, model, prompt)
        elif provider == "openai":
            return generate_openai(api_key, model, prompt)
        elif provider == "anthropic":
            return generate_anthropic(api_key, model, prompt)
        elif provider == "ollama":
            return generate_ollama(ollama_url, model, prompt)
        else:
            return None
    except Exception as e:
        log_message(f"Erreur génération IA: {str(e)}", "ERROR")
        return None

def generate_gemini(api_key, model, prompt):
    """Générer avec Gemini"""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent?alt=sse&key={api_key}"
    
    data = {
        "contents": [{
            "parts": [{
                "text": prompt
            }]
        }]
    }
    
    response = requests.post(url, json=data, timeout=30, stream=True)
    response.raise_for_status()
    
    # Parser la réponse SSE (Server-Sent Events)
    full_text = ""
    for line in response.iter_lines():
        if line:
            line_str = line.decode('utf-8')
            if line_str.startswith('data: '):
                json_str = line_str[6:]  # Enlever 'data: '
                if json_str.strip() == '[DONE]':
                    break
                try:
                    chunk = json.loads(json_str)
                    if 'candidates' in chunk and len(chunk['candidates']) > 0:
                        text = chunk['candidates'][0]['content']['parts'][0].get('text', '')
                        full_text += text
                except json.JSONDecodeError:
                    continue
    
    return full_text if full_text else None

def generate_openai(api_key, model, prompt):
    """Générer avec OpenAI"""
    url = "https://api.openai.com/v1/chat/completions"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    
    response = requests.post(url, headers=headers, json=data, timeout=30)
    response.raise_for_status()
    
    result = response.json()
    return result['choices'][0]['message']['content']

def generate_anthropic(api_key, model, prompt):
    """Générer avec Anthropic"""
    url = "https://api.anthropic.com/v1/messages"
    
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01"
    }
    
    data = {
        "model": model,
        "max_tokens": 1024,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    
    response = requests.post(url, headers=headers, json=data, timeout=30)
    response.raise_for_status()
    
    result = response.json()
    return result['content'][0]['text']

def generate_ollama(ollama_url, model, prompt):
    """Générer avec Ollama"""
    url = f"{ollama_url}/api/generate"
    
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    
    response = requests.post(url, json=data, timeout=60)
    response.raise_for_status()
    
    result = response.json()
    return result.get('response', '')

def send_email_response(config, to_address, subject, body_text, in_reply_to=None, references=None):
    """Envoyer une réponse par email"""
    try:
        msg = MIMEMultipart('alternative')
        msg['From'] = config['email_address']
        msg['To'] = to_address
        msg['Subject'] = f"Re: {subject}" if not subject.startswith('Re:') else subject
        
        if in_reply_to:
            msg['In-Reply-To'] = in_reply_to
        if references:
            msg['References'] = references
        
        # Template Nørd
        html_body = f"""<!DOCTYPE html>
<html>
<head>
    <meta http-equiv="Content-Type" content="text/html; charset=utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Réponse automatique - Nørd</title>
    <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=Playfair+Display:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'DM Sans', sans-serif;
            background-color: #0a0e0d;
            color: #f0ede6;
            line-height: 1.6;
        }}
        
        .email-container {{
            max-width: 600px;
            margin: 0 auto;
            background-color: #0a0e0d;
        }}
        
        .header {{
            background: linear-gradient(135deg, #5daa7f 0%, #4a8f6a 100%);
            padding: 40px 30px;
            text-align: center;
        }}
        
        .logo {{
            font-family: 'Playfair Display', serif;
            font-size: 36px;
            font-weight: 700;
            color: #0a0e0d;
            margin-bottom: 10px;
            letter-spacing: -1px;
        }}
        
        .logo-subtitle {{
            font-size: 14px;
            color: rgba(10, 14, 13, 0.8);
            font-weight: 500;
        }}
        
        .content {{
            padding: 50px 30px;
            background-color: #0a0e0d;
        }}
        
        .ai-response {{
            font-size: 16px;
            color: #f0ede6;
            line-height: 1.8;
            margin-bottom: 30px;
        }}
        
        .signature {{
            margin-top: 40px;
            padding-top: 30px;
            border-top: 1px solid rgba(93, 170, 127, 0.3);
        }}
        
        .signature-name {{
            font-family: 'Playfair Display', serif;
            font-size: 20px;
            font-weight: 600;
            color: #5daa7f;
            margin-bottom: 10px;
        }}
        
        .ai-badge {{
            display: inline-block;
            background: linear-gradient(135deg, rgba(93, 170, 127, 0.2) 0%, rgba(45, 74, 66, 0.2) 100%);
            border: 1px solid rgba(93, 170, 127, 0.4);
            color: #5daa7f;
            padding: 8px 16px;
            border-radius: 8px;
            font-size: 13px;
            margin-top: 15px;
            font-weight: 500;
        }}
        
        .footer {{
            background-color: #1a2825;
            padding: 30px;
            text-align: center;
            border-top: 1px solid rgba(93, 170, 127, 0.2);
            margin-top: 50px;
        }}
        
        .footer p {{
            font-size: 13px;
            color: rgba(240, 237, 230, 0.7);
            margin: 8px 0;
        }}
        
        @media (max-width: 600px) {{
            .header {{
                padding: 30px 20px;
            }}
            
            .content {{
                padding: 30px 20px;
            }}
            
            .logo {{
                font-size: 28px;
            }}
        }}
    </style>
</head>
<body>
    <div class="email-container">
        <div class="header">
            <div class="logo">NØRD</div>
            <div class="logo-subtitle">Créé par Lynx</div>
        </div>
        
        <div class="content">
            <div class="ai-response">
                {body_text.replace(chr(10), '<br>')}
            </div>
            
            <div class="signature">
                <div class="signature-name">{config['email_address']}</div>
                <div class="ai-badge">✦ Réponse générée par IA • {config['ai_provider'].title()} ({config['ai_model']})</div>
            </div>
        </div>
        
        <div class="footer">
            <p><strong>NØRD — Intelligence Artificielle</strong></p>
            <p>Une IA utile, sobre et humaine</p>
        </div>
    </div>
</body>
</html>"""
        
        part_text = MIMEText(body_text, 'plain', 'utf-8')
        part_html = MIMEText(html_body, 'html', 'utf-8')
        msg.attach(part_text)
        msg.attach(part_html)
        
        with smtplib.SMTP(config['smtp_server'], config['smtp_port']) as server:
            server.starttls()
            server.login(config['email_address'], config['email_password'])
            server.send_message(msg)
        
        log_message(f"Email envoyé avec succès à {to_address}", "SUCCESS")
        monitoring_state['stats']['emails_responded'] += 1
        return True
    except Exception as e:
        log_message(f"Erreur envoi email: {str(e)}", "ERROR")
        monitoring_state['stats']['errors'] += 1
        return False

def process_email(imap, config, uid):
    """Traiter un email spécifique"""
    try:
        _, msg_data = imap.uid('fetch', uid, '(RFC822)')
        email_body = msg_data[0][1]
        msg = email.message_from_bytes(email_body)
        
        subject = decode_header_value(msg.get('Subject', ''))
        from_header = decode_header_value(msg.get('From', ''))
        message_id = msg.get('Message-ID', '')
        references = msg.get('References', '')
        
        sender_email = email.utils.parseaddr(from_header)[1]
        
        if config['keyword'].lower() in subject.lower():
            log_message(f"Email détecté: {subject} de {sender_email}", "INFO")
            
            body = get_email_body(msg)
            
            log_message(f"Génération de réponse IA ({config['ai_provider']}/{config['ai_model']})...", "INFO")
            
            ai_response = generate_ai_response(
                config['ai_provider'],
                config.get('api_key', ''),
                config['ai_model'],
                config.get('ollama_url', 'http://localhost:11434'),
                config.get('custom_context', ''),
                sender_email,
                subject,
                body
            )
            
            if ai_response:
                log_message("Réponse IA générée avec succès", "SUCCESS")
                success = send_email_response(
                    config,
                    sender_email,
                    subject,
                    ai_response,
                    in_reply_to=message_id,
                    references=references
                )
                
                if success:
                    log_message(f"Email traité avec succès: {subject}", "SUCCESS")
            else:
                log_message("Aucune réponse IA générée", "ERROR")
                
    except Exception as e:
        log_message(f"Erreur traitement email: {str(e)}", "ERROR")
        monitoring_state['stats']['errors'] += 1

def monitoring_loop():
    """Boucle de surveillance"""
    config = monitoring_state['config']
    imap = None
    check_count = 0
    
    log_message("Démarrage de la surveillance des emails", "SUCCESS")
    monitoring_state['stats']['start_time'] = datetime.now().isoformat()
    
    while monitoring_state['running']:
        try:
            if not imap:
                log_message(f"Connexion au serveur IMAP {config['imap_server']}:{config['imap_port']}", "INFO")
                imap = imaplib.IMAP4_SSL(config['imap_server'], config['imap_port'])
                imap.login(config['email_address'], config['email_password'])
                log_message("Connexion IMAP réussie", "SUCCESS")
            
            imap.select('INBOX')
            
            _, message_numbers = imap.uid('search', None, 'UNSEEN')
            
            check_count += 1
            
            if message_numbers[0]:
                uids = message_numbers[0].split()
                log_message(f"{len(uids)} email(s) non lu(s) détecté(s)", "INFO")
                
                for uid in uids:
                    if uid not in monitoring_state['processed_uids']:
                        monitoring_state['stats']['emails_scanned'] += 1
                        process_email(imap, config, uid)
                        monitoring_state['processed_uids'].add(uid)
            else:
                if check_count % 5 == 0:
                    log_message(f"Vérification #{check_count} - Aucun nouveau message", "INFO")
            
            current_interval = config.get('check_interval', 30)
            time.sleep(current_interval)
            
        except imaplib.IMAP4.abort:
            log_message("Connexion IMAP interrompue, reconnexion...", "WARNING")
            imap = None
            time.sleep(5)
        except Exception as e:
            log_message(f"Erreur dans la boucle de surveillance: {str(e)}", "ERROR")
            monitoring_state['stats']['errors'] += 1
            time.sleep(10)
    
    if imap:
        try:
            imap.logout()
        except:
            pass
    
    log_message("Surveillance arrêtée", "INFO")

# Routes API

@app.route('/')
def index():
    """Servir la page HTML"""
    return send_from_directory('.', 'index.html')

@app.route('/api/config', methods=['GET', 'POST'])
def config_endpoint():
    """Gérer la configuration"""
    if request.method == 'POST':
        monitoring_state['config'] = request.json
        return jsonify({'success': True})
    else:
        return jsonify(monitoring_state['config'])

@app.route('/api/start', methods=['POST'])
def start_monitoring():
    """Démarrer la surveillance"""
    if monitoring_state['running']:
        return jsonify({'success': False, 'error': 'Already running'})
    
    monitoring_state['running'] = True
    monitoring_state['processed_uids'] = set()
    monitoring_state['stats'] = {
        'emails_scanned': 0,
        'emails_responded': 0,
        'errors': 0,
        'start_time': None
    }
    monitoring_state['logs'] = []
    
    thread = threading.Thread(target=monitoring_loop, daemon=True)
    thread.start()
    monitoring_state['thread'] = thread
    
    return jsonify({'success': True})

@app.route('/api/stop', methods=['POST'])
def stop_monitoring():
    """Arrêter la surveillance"""
    monitoring_state['running'] = False
    return jsonify({'success': True})

@app.route('/api/status', methods=['GET'])
def get_status():
    """Obtenir le statut actuel"""
    return jsonify({
        'running': monitoring_state['running'],
        'stats': monitoring_state['stats'],
        'logs': monitoring_state['logs'][-50:]  # Derniers 50 logs
    })

@app.route('/api/test-ia', methods=['POST'])
def test_ia():
    """Tester la configuration IA"""
    data = request.json
    
    try:
        response = generate_ai_response(
            data['provider'],
            data.get('api_key', ''),
            data['model'],
            data.get('ollama_url', 'http://localhost:11434'),
            '',
            'test@example.com',
            'Test',
            'Réponds simplement "Bonjour!" pour confirmer que tu fonctionnes.'
        )
        
        if response:
            return jsonify({'success': True, 'response': response})
        else:
            return jsonify({'success': False, 'error': 'No response generated'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    print("🚀 Email AI Auto-Responder Backend démarré")
    print("📡 Serveur: http://localhost:5000")
    print("🌐 Ouvrez http://localhost:5000 dans votre navigateur")
    app.run(debug=True, host='0.0.0.0', port=5000)