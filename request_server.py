from pathlib import Path
import sys
import io
import base64
import json
import requests
from pprint import pprint

if __name__ == "__main__":
    folder = sys.argv[1]
    base_url = sys.argv[2]
    access_token = sys.argv[3]
    request_url = f"{base_url}/upload?password={access_token}"
    
    print(f"Looping through: {folder}")
    for image_file in Path(folder).glob("*.png"):
        with open(image_file, "rb") as f:
            im_bytes = f.read()        
        im_b64 = base64.b64encode(im_bytes).decode("utf8")
        files = {'file': open(image_file, 'rb')}
        headers = {"password": access_token}
        payload = json.dumps({"file": im_b64})
        response = requests.post(
            request_url,
            files=files
        )

        pprint(f"{image_file}:") 
        pprint(response.json())