from BaseHTTPServer import HTTPServer, BaseHTTPRequestHandler
from urlparse import parse_qs, urlparse

class SpotHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        url = urlparse(self.path)
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(parse_qs((url)['videoId']))
    

PORT = 8081
HOST_NAME = ''

if __name__ == '__main__':
    HTTPServer((HOST_NAME, PORT), SpotHandler).serve_forever()
