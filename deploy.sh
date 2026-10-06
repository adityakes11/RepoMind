#!/bin/bash
# ── RepoMind Deployment Script ────────────────────────────────────
# Run this on your EC2 instance after cloning the repo
set -euo pipefail

echo "══════════════════════════════════════════════"
echo "  RepoMind – EC2 Deployment"
echo "══════════════════════════════════════════════"

# ── 1. Check prerequisites ───────────────────────────────────────
echo ""
echo "[1/6] Checking prerequisites..."

for cmd in docker git; do
    if ! command -v "$cmd" &>/dev/null; then
        echo "❌ $cmd not found. Please install it first."
        exit 1
    fi
done

# Check docker compose (v2 plugin or standalone)
if docker compose version &>/dev/null; then
    COMPOSE="docker compose"
elif command -v docker-compose &>/dev/null; then
    COMPOSE="docker-compose"
else
    echo "❌ docker compose not found. Install it:"
    echo "   sudo apt install docker-compose-plugin"
    exit 1
fi

echo "✅ docker: $(docker --version)"
echo "✅ compose: $($COMPOSE version)"

# ── 2. Check PostgreSQL ──────────────────────────────────────────
echo ""
echo "[2/6] Checking PostgreSQL..."

if command -v psql &>/dev/null; then
    echo "✅ PostgreSQL client found"
else
    echo "⚠️  psql not found (PostgreSQL may still be running)"
fi

# ── 3. Check Ollama ──────────────────────────────────────────────
echo ""
echo "[3/6] Checking Ollama..."

if curl -s http://localhost:11434/api/tags &>/dev/null; then
    echo "✅ Ollama is running"
else
    echo "❌ Ollama not responding on localhost:11434"
    echo "   Start it with: ollama serve"
    exit 1
fi

# ── 4. Configure PostgreSQL for Docker ───────────────────────────
echo ""
echo "[4/6] Configuring PostgreSQL to accept Docker connections..."

# Get Docker bridge IP
DOCKER_IP=$(docker network inspect bridge --format '{{range .IPAM.Config}}{{.Gateway}}{{end}}' 2>/dev/null || echo "172.17.0.1")

echo "   Docker bridge gateway: $DOCKER_IP"
echo ""
echo "   ⚠️  Make sure PostgreSQL accepts connections from Docker:"
echo "   1. Edit /etc/postgresql/*/main/postgresql.conf"
echo "      → listen_addresses = '*'"
echo "   2. Edit /etc/postgresql/*/main/pg_hba.conf"
echo "      → Add: host all all 172.17.0.0/16 md5"
echo "   3. Restart: sudo systemctl restart postgresql"
echo ""
read -p "   Have you done this? (y/n): " PG_READY
if [[ "$PG_READY" != "y" ]]; then
    echo "   Please configure PostgreSQL first, then re-run this script."
    exit 1
fi

# ── 5. Create the database ───────────────────────────────────────
echo ""
echo "[5/6] Creating database if needed..."
sudo -u postgres psql -tc "SELECT 1 FROM pg_database WHERE datname = 'repomind'" | grep -q 1 \
    || sudo -u postgres psql -c "CREATE DATABASE repomind"
echo "✅ Database 'repomind' ready"

# ── 6. Setup .env ────────────────────────────────────────────────
echo ""
echo "[6/6] Setting up environment..."

if [ ! -f .env ]; then
    cp .env.production .env
    echo "✅ Created .env from .env.production"
    echo "   ⚠️  Edit .env and set your POSTGRES_PASSWORD"
    echo "   Then run: $COMPOSE up -d --build"
else
    echo "✅ .env already exists"
fi

# ── Build and Launch ─────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════"
echo "  Building and starting containers..."
echo "══════════════════════════════════════════════"

$COMPOSE up -d --build

echo ""
echo "══════════════════════════════════════════════"
echo "  ✅ RepoMind is LIVE!"
echo ""
echo "  Frontend:  http://$(curl -s ifconfig.me):80"
echo "  API:       http://$(curl -s ifconfig.me):8000"
echo "  Swagger:   http://$(curl -s ifconfig.me):8000/docs"
echo "  Health:    http://$(curl -s ifconfig.me):8000/api/health"
echo "══════════════════════════════════════════════"

