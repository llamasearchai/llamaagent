# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

## Reporting a Vulnerability

We take the security of LlamaAgent seriously. If you have discovered a security vulnerability, please follow these steps:

1. **DO NOT** create a public GitHub issue for the vulnerability.
2. Email your findings to nikjois@llamasearch.ai
3. Include the following in your report:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

## Response Timeline

- We will acknowledge receipt of your vulnerability report within 48 hours
- We will provide a detailed response within 7 days
- We will work on a fix and coordinate disclosure with you

## Security Best Practices

When using LlamaAgent:

1. **API Keys**: Never commit API keys to version control
2. **Authentication**: Always use JWT tokens in production
3. **Rate Limiting**: Configure appropriate rate limits
4. **Input Validation**: Enable all security validators
5. **Monitoring**: Set up security monitoring and alerts

## Security Features

LlamaAgent includes:
- JWT authentication
- Rate limiting
- Input validation
- Audit logging
- Encrypted storage options

Thank you for helping keep LlamaAgent secure!