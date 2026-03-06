import urllib.request
import urllib.parse
import mimetypes

url = "http://localhost:5000/admin/documents/extract"
file_path = "test_doc.txt"

with open(file_path, "rb") as f:
    file_data = f.read()

boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
body = (
    f"--{boundary}\r\n"
    f'Content-Disposition: form-data; name="file"; filename="test_doc.txt"\r\n'
    f"Content-Type: text/plain\r\n\r\n"
).encode('utf-8') + file_data + f"\r\n--{boundary}--\r\n".encode('utf-8')

req = urllib.request.Request(url, data=body)
req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")

try:
    with urllib.request.urlopen(req) as response:
        print("Status Code:", response.getcode())
        print("Response JSON:", response.read().decode('utf-8'))
except Exception as e:
    print("Error:", e)
