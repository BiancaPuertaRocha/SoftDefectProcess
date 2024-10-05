import random
import requests
import string

def generate_random_token_name(base_name):
    """Generates a new random name for the token."""
    suffix = ''.join(random.choices(string.ascii_letters + string.digits, k=5))
    return f"{base_name}_{suffix}"


def generate_new_sonar_token(sonar_url, sonar_user, sonar_pass):
    """Generates a new auth token."""
    token = None
    token_name = 'sonnar_token_renewed'
    
    print("Generating a new SonarQube token...")
    
    while True:
        response = requests.post(
            f'{sonar_url}/api/user_tokens/generate',
            auth=(sonar_user, sonar_pass),
            data={'name': token_name}
        )
        
        if response.status_code == 200:
            token = response.json()['token']
            print(f"New Token: {token}")
            return token
        elif "already exists" in response.text:
            print(f"The token {token_name} already exists.Generating new...")
            token_name = generate_random_token_name(token_name)
        else:
            print(f"Erro ao gerar um novo token: {response.text}")
            return None
        

