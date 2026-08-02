#!/bin/sh
set -e

echo "================================="
echo "Running database migrations..."
echo "================================="

python manage.py migrate --noinput

echo "================================="
echo "Migrations completed."
echo "Starting server..."
echo "================================="

exec gunicorn newsbot_web.wsgi:application \
    --bind 0.0.0.0:${PORT}