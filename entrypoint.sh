#!/bin/bash
chmod 600 /root/.ssh/id_rsa
chmod 700 /root/.ssh
exec "$@"