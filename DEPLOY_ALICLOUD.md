# Deploying Stock Backtest System to Alibaba Cloud

## Prerequisites
1. Alibaba Cloud ECS instance (Ubuntu/Debian)
2. SSH access to the server
3. Domain name: openquant.club (already acquired)
4. Basic Linux command knowledge

## Deployment Steps

### 1. Prepare Your Local Code
First, ensure your code is ready and committed:
```bash
cd /home/tira/code/stock_backtest
git add -A
git commit -m "Prepare for Alibaba Cloud deployment"
```

### 2. Transfer Code to Alibaba Cloud Server
You can use scp or rsync to transfer your code:

```bash
# Using scp (replace with your server's public IP)
scp -r /home/tira/code/stock_backtest root@your_server_ip:/opt/

# Or using rsync
rsync -avz /home/tira/code/stock_backtest/ root@your_server_ip:/opt/stock_backtest/
```

### 3. Connect to Your Alibaba Cloud Server
```bash
ssh root@your_server_ip
```

### 4. Install Required Dependencies
```bash
# Update system packages
apt update && apt upgrade -y

# Install Docker and Docker Compose
apt install -y docker.io docker-compose

# Start and enable Docker service
systemctl start docker
systemctl enable docker

# Add your user to docker group (if not root)
usermod -aG docker $USER
newgrp docker
```

### 5. Navigate to the Project Directory
```bash
cd /opt/stock_backtest
```

### 6. Build and Run the Application
```bash
# Build Docker image
docker-compose build

# Start the application
docker-compose up -d

# Check if containers are running
docker-compose ps
```

### 7. Verify Deployment
Once the containers are running, you can access the application:
- Using IP address: Open a web browser and navigate to: `http://your_server_ip`
- Using domain name: Navigate to: `http://openquant.club`
- You should see the stock backtest system interface
- API documentation will be available at: `http://openquant.club/docs` or `http://your_server_ip/docs`

### 8. Configure Firewall (Security Group)
In Alibaba Cloud console:
1. Go to your ECS instance's security group
2. Add inbound rule to allow traffic on port 80 (HTTP)
   - Port range: 80/80
   - Authorization object: 0.0.0.0/0 (or restrict to your IP for security)
   - Policy: Allow

### 9. Optional: Set Up Domain Name
If you have a domain name:
1. Point your domain's A record to your server's public IP
2. Wait for DNS propagation (usually minutes to hours)
3. Access your application via your domain name

### 10. Optional: Set Up HTTPS with Let's Encrypt and Nginx (Recommended for Production)
For secure HTTPS access using your domain name with proper SSL termination:

#### Option A: Using Nginx Reverse Proxy (Recommended)
1. Install Nginx and Certbot:
```bash
apt install -y nginx certbot python3-certbot-nginx
```

2. Obtain SSL certificate:
```bash
certbot --nginx -d openquant.club -d www.openquant.club
```

3. Configure Nginx as reverse proxy (uses the nginx.conf provided):
```bash
# Copy the nginx configuration
cp nginx.conf /etc/nginx/sites-available/openquant.club

# Enable the site
ln -s /etc/nginx/sites-available/openquant.club /etc/nginx/sites-enabled/

# Test and reload Nginx
nginx -t
systemctl reload nginx
```

4. Update docker-compose.yml to bind only to localhost (since Nginx handles external traffic):
```yaml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "127.0.0.1:8000:8000"  # Only accessible locally
    restart: unless-stopped
```

#### Option B: Direct Container Ports (Simpler but less secure)
If you prefer not to use Nginx:
```bash
# Install certbot for standalone certificate
apt install -y certbot

# Obtain SSL certificate (stop containers first if using ports 80/443)
docker-compose down
certbot certonly --standalone -d openquant.club -d www.openquant.club
docker-compose up -d

# Update docker-compose.yml to mount certificates
version: '3.8'

services:
  web:
    build: .
    ports:
      - "80:8000"
      - "443:8000"
    volumes:
      - /etc/letsencrypt:/etc/letsencrypt:ro
    restart: unless-stopped
```

Note: Option B requires your application to handle HTTPS termination directly, which it currently doesn't do. Option A with Nginx is recommended for proper SSL handling.

### 10. Maintenance Commands
```bash
# View logs
docker-compose logs -f

# Stop the application
docker-compose down

# Restart the application
docker-compose restart

# Update code and redeploy
# (After pulling new code on server)
docker-compose build
docker-compose up -d
```

## Troubleshooting
- If you see port conflicts, ensure no other service is using port 80
- Check Docker logs: `docker-compose logs`
- Verify container status: `docker-compose ps`
- Ensure your Alibaba Cloud security group allows inbound traffic on port 80

## Security Considerations
1. Consider setting up HTTPS with Let's Encrypt for production use
2. Restrict access to specific IPs in security group if needed
3. Regularly update Docker images and dependencies
4. Consider setting up automatic backups of your data

## Notes
- The application runs on port 8000 inside the container but is mapped to port 80 on the host
- No database is configured by default - the app will run in SQLite/file-based mode
- For production use with persistent data, consider configuring DATABASE_URL for PostgreSQL