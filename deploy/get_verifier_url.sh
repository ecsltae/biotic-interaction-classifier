#!/bin/bash
# Print the current live Cloudflare tunnel URL for the verifier.
# Quick tunnels change hostname on every restart, so read it here rather than hardcoding it.
journalctl -u metap-verifier-tunnel --no-pager | grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' | tail -1
