#!/bin/bash
# Generate self-signed SSL certificates for LlamaAgent development/testing
# Author: Nik Jois <nikjois@llamasearch.ai>
#
# For production, replace these with proper certificates from a CA

set -e

# Certificate configuration
CERT_DIR="$(dirname "$0")"
CERT_NAME="llamaagent"
CERT_DAYS=365
KEY_SIZE=2048

# Certificate details
COUNTRY="US"
STATE="California"
CITY="San Francisco"
ORG="LlamaAgent"
OU="Development"
CN="llamaagent.local"

# Subject Alternative Names
SAN="DNS:llamaagent.local,DNS:api.llamaagent.local,DNS:monitoring.llamaagent.local,DNS:localhost,IP:127.0.0.1"

echo "Generating SSL certificates for LlamaAgent..."

# Generate private key
openssl genrsa -out "$CERT_DIR/key.pem" $KEY_SIZE

# Generate certificate signing request
openssl req -new -key "$CERT_DIR/key.pem" -out "$CERT_DIR/cert.csr" -subj "/C=$COUNTRY/ST=$STATE/L=$CITY/O=$ORG/OU=$OU/CN=$CN"

# Generate self-signed certificate with SAN
openssl x509 -req -in "$CERT_DIR/cert.csr" -signkey "$CERT_DIR/key.pem" -out "$CERT_DIR/cert.pem" -days $CERT_DAYS -extensions v3_req -extfile <(
cat << EOF
[v3_req]
basicConstraints = CA:FALSE
keyUsage = nonRepudiation, digitalSignature, keyEncipherment
subjectAltName = $SAN
EOF
)

# Create certificate chain (self-signed, so same as cert)
cp "$CERT_DIR/cert.pem" "$CERT_DIR/chain.pem"

# Set proper permissions
chmod 600 "$CERT_DIR/key.pem"
chmod 644 "$CERT_DIR/cert.pem" "$CERT_DIR/chain.pem"

# Clean up CSR
rm "$CERT_DIR/cert.csr"

echo "SSL certificates generated successfully!"
echo "Certificate: $CERT_DIR/cert.pem"
echo "Private key: $CERT_DIR/key.pem"
echo "Chain: $CERT_DIR/chain.pem"
echo ""
echo "For development, add these domains to your /etc/hosts:"
echo "127.0.0.1 llamaagent.local"
echo "127.0.0.1 api.llamaagent.local"
echo "127.0.0.1 monitoring.llamaagent.local"