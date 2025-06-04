#!/bin/bash

# Trading Strategy Optimization API - Monitoring Setup Script
# This script sets up Grafana and Prometheus for monitoring the API

set -e

echo "🚀 Setting up Trading Strategy Optimization API Monitoring Stack"
echo "================================================================"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker or OrbStack first."
    echo "   OrbStack is recommended for macOS: https://orbstack.dev"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

echo "📂 Working directory: $SCRIPT_DIR"

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p grafana/provisioning/datasources
mkdir -p grafana/provisioning/dashboards
mkdir -p grafana/dashboards

# Check if API is running
echo "🔍 Checking if Trading Strategy API is running..."
if curl -s http://localhost:8000/api/v1/health > /dev/null; then
    echo "✅ API is running on http://localhost:8000"
else
    echo "⚠️  API is not running on http://localhost:8000"
    echo "   Please start the API first with: cd ../.. && python run_server.py"
    echo "   Continuing with setup anyway..."
fi

# Stop any existing containers
echo "🛑 Stopping any existing monitoring containers..."
docker-compose down 2>/dev/null || true

# Start the monitoring stack
echo "🚀 Starting monitoring stack..."
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 10

# Check Prometheus
echo "🔍 Checking Prometheus..."
if curl -s http://localhost:9090/-/healthy > /dev/null; then
    echo "✅ Prometheus is running on http://localhost:9090"
else
    echo "❌ Prometheus failed to start"
fi

# Check Grafana
echo "🔍 Checking Grafana..."
if curl -s http://localhost:3001/api/health > /dev/null; then
    echo "✅ Grafana is running on http://localhost:3001"
else
    echo "❌ Grafana failed to start"
fi

echo ""
echo "🎉 Monitoring Stack Setup Complete!"
echo "=================================="
echo ""
echo "📊 Access Points:"
echo "  • Grafana Dashboard: http://localhost:3001"
echo "    - Username: admin"
echo "    - Password: trading_api_2024"
echo ""
echo "  • Prometheus: http://localhost:9090"
echo "  • Node Exporter: http://localhost:9100"
echo ""
echo "  • Trading API Dashboard: http://localhost:8000/api/v1/monitoring/"
echo "    - Requires API key: dev_key_123"
echo ""
echo "💡 OrbStack Users:"
echo "  • This monitoring stack works seamlessly with OrbStack"
echo "  • OrbStack's efficient networking provides excellent performance"
echo "  • Check OrbStack menu bar for easy container management"
echo ""
echo "📈 Available Dashboards:"
echo "  • Trading Strategy Optimization API - Overview"
echo "    - API performance metrics"
echo "    - System resource monitoring"
echo "    - Optimization job tracking"
echo "    - Error rates and health status"
echo ""
echo "🔧 Management Commands:"
echo "  • View logs: docker-compose logs -f"
echo "  • Stop stack: docker-compose down"
echo "  • Restart: docker-compose restart"
echo "  • Update: docker-compose pull && docker-compose up -d"
echo ""
echo "📝 Notes:"
echo "  • Prometheus data retention: 200 hours"
echo "  • Grafana data persisted in Docker volumes"
echo "  • API metrics scraped every 30 seconds"
echo "  • Dashboards auto-refresh every 30 seconds" 