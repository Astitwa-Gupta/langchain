import requests
import base64
import json

class VeevaVaultClient:
    def __init__(self, username, password, client_id, client_secret, base_url):
        self.username = username
        self.password = password
        self.client_id = client_id
        self.client_secret = client_secret
        self.base_url = base_url.rstrip('/')
        self.access_token = None
        self.session = requests.Session()
        
    def authenticate(self):
        """Authenticate and get access token"""
        auth_url = f"{self.base_url}/api/v20.2/auth"
        
        # Prepare authentication headers
        auth_string = f"{self.client_id}:{self.client_secret}"
        encoded_auth = base64.b64encode(auth_string.encode()).decode()
        
        headers = {
            'Authorization': f'Basic {encoded_auth}',
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        
        # Prepare authentication data
        data = {
            'username': self.username,
            'password': self.password,
            'grant_type': 'password'
        }
        
        try:
            response = self.session.post(auth_url, headers=headers, data=data)
            response.raise_for_status()
            
            auth_data = response.json()
            self.access_token = auth_data.get('access_token')
            
            if self.access_token:
                # Set authorization header for future requests
                self.session.headers.update({
                    'Authorization': f'Bearer {self.access_token}'
                })
                return True
            else:
                raise Exception("Authentication failed: No access token received")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"Authentication error: {str(e)}")
			






    def get_document_by_id(self, document_id):
        """Get document metadata by ID"""
        doc_url = f"{self.base_url}/api/v20.2/objects/documents/{document_id}"
        
        try:
            response = self.session.get(doc_url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error fetching document: {str(e)}")
    
    def download_document_file(self, document_id, output_path=None):
        """Download document file by ID"""
        # First get document metadata to check file details
        doc_metadata = self.get_document_by_id(document_id)
        
        # Get the major version for download
        major_version = doc_metadata.get('major_version_number', 0)
        
        # Construct download URL
        download_url = f"{self.base_url}/api/v20.2/objects/documents/{document_id}/file"
        
        try:
            response = self.session.get(download_url, stream=True)
            response.raise_for_status()
            
            # Determine output filename
            if output_path is None:
                # Use document name from metadata
                doc_name = doc_metadata.get('name', f'document_{document_id}')
                output_path = f"{doc_name}.pdf"  # Adjust extension as needed
            
            # Save the file
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return output_path
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error downloading file: {str(e)}")
			
			
			
			
			
def main():
    # Configuration - replace with your actual credentials
    config = {
        'username': 'your_username',
        'password': 'your_password',
        'client_id': 'your_client_id',
        'client_secret': 'your_client_secret',
        'base_url': 'https://your-veeva-vault-domain.veevavault.com'
    }
    
    # Document ID to fetch
    document_id = 'your_document_id_here'
    
    # Initialize client
    client = VeevaVaultClient(**config)
    
    try:
        # Authenticate
        if client.authenticate():
            print("Authentication successful!")
            
            # Fetch document metadata
            metadata = client.get_document_by_id(document_id)
            print(f"Document Name: {metadata.get('name')}")
            print(f"Document Type: {metadata.get('type__v')}")
            
            # Download the file
            output_file = client.download_document_file(document_id, f"downloaded_{document_id}.pdf")
            print(f"File downloaded successfully: {output_file}")
            
        else:
            print("Authentication failed!")
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
