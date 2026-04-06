#!/usr/bin/env bash
# One-time Ubuntu VPS setup for cpr-assist backend (run as ubuntu with sudo).
#
#   export ASSIST_ROOT=/opt/cpr-assist    # root of this repo clone
#   export SSLIP_HOST=141-94-78-60.sslip.io   # optional; omit to skip nginx
#   bash infra/vps/bootstrap.sh
#
# Then: copy backend/.env (JWT_SECRET, MONGODB_URI, …).

set -euo pipefail

ASSIST_ROOT="${ASSIST_ROOT:-/opt/cpr-assist}"
BACKEND="${ASSIST_ROOT}/backend"
SSLIP_HOST="${SSLIP_HOST:-}"

if [[ ! -f "${BACKEND}/pyproject.toml" ]]; then
  echo "Missing ${BACKEND}/pyproject.toml — set ASSIST_ROOT to your cpr-assist clone"
  exit 1
fi

sudo apt-get update -y
sudo apt-get install -y python3-venv python3-pip nginx git

# Model weights under cpr_ml/experiments/ are not in Git — copy with scp/rsync (see cpr_ml/README.md).
if [[ ! -f "${ASSIST_ROOT}/cpr_ml/experiments/cpr_s0_image_classifier/runs/s0_image_model/best.pt" ]]; then
  echo "Note: S0 checkpoint missing at cpr_ml/experiments/.../best.pt — copy weights before relying on full vision."
fi

python3 -m venv "${BACKEND}/.venv"
# shellcheck source=/dev/null
source "${BACKEND}/.venv/bin/activate"
pip install -U pip
pip install -e "${BACKEND}[vision]"

sudo tee /etc/systemd/system/cpr-assist.service >/dev/null <<EOF
[Unit]
Description=CPR Assist FastAPI
After=network.target

[Service]
User=ubuntu
Group=ubuntu
WorkingDirectory=${BACKEND}
Environment=PATH=${BACKEND}/.venv/bin
ExecStart=${BACKEND}/.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now cpr-assist

if [[ -n "${SSLIP_HOST}" ]]; then
  sudo tee /etc/nginx/sites-available/cpr-assist >/dev/null <<EOF
server {
    listen 80;
    server_name ${SSLIP_HOST};

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF
  sudo ln -sf /etc/nginx/sites-available/cpr-assist /etc/nginx/sites-enabled/cpr-assist
  sudo rm -f /etc/nginx/sites-enabled/default
  sudo nginx -t
  sudo systemctl reload nginx
  echo "Install TLS: sudo apt install -y certbot python3-certbot-nginx && sudo certbot --nginx -d ${SSLIP_HOST}"
else
  echo "Skipped nginx (no SSLIP_HOST)."
fi

echo "Done. Check: curl -sS http://127.0.0.1:8000/health"
echo "GitHub: ENABLE_VPS_DEPLOY, VPS_* secrets; VPS_DEPLOY_PATH=${ASSIST_ROOT}"
