const http = require('http');
const fs = require('fs');
const url = require('url');
const qs = require('querystring');

// Create a server
const server = http.createServer((req, res) => {
    // Parse the URL of the request
    const parsedUrl = url.parse(req.url, true);

    // Check if the request is for storing data
    if (parsedUrl.pathname === '/store-data' && req.method === 'POST') {
        // Initialize a string to store the data
        let data = '';

        // Append incoming data to the string
        req.on('data', (chunk) => {
            data += chunk;
        });

        // When all data is received
        req.on('end', () => {
            // Parse the query string to get the option value
            const postData = qs.parse(data);

            // Extract the option value
            const option = postData.option;

            // Write the option to a CSV file
            fs.appendFile('data.csv', option + '\n', (err) => {
                if (err) throw err;
                console.log('Data written to CSV file');
            });

            // Send a response back to the client
            res.writeHead(200, { 'Content-Type': 'text/plain' });
            res.end('Data stored successfully');
        });
    } else {
        // For other requests, serve the HTML file
        fs.readFile('index.html', (err, data) => {
            if (err) {
                res.writeHead(404, { 'Content-Type': 'text/plain' });
                res.end('Error: File not found');
            } else {
                res.writeHead(200, { 'Content-Type': 'text/html' });
                res.end(data);
            }
        });
    }
});

// Start the server
const PORT = process.env.PORT || 3000;
server.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});
