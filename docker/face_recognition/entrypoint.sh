#!/bin/bash
set -e

# Wait for database to be available
echo "Waiting for PostgreSQL to be available at $POSTGRES_HOST:$POSTGRES_PORT..."
timeout=120  # Increase timeout to 2 minutes

# Install the PostgreSQL client if not already present
if ! [ -x "$(command -v pg_isready)" ]; then
  echo "Installing PostgreSQL client"
  apt-get update && apt-get install -y postgresql-client
fi

# Try connecting to PostgreSQL
while ! pg_isready -h $POSTGRES_HOST -p $POSTGRES_PORT -U $POSTGRES_USER -t 1 >/dev/null 2>&1
do
    timeout=$(expr $timeout - 1)
    if [ $timeout -eq 0 ]; then
        echo "Timeout waiting for PostgreSQL to be available"
        echo "Diagnostic information:"
        echo "PostgreSQL host: $POSTGRES_HOST"
        echo "PostgreSQL port: $POSTGRES_PORT"
        echo "PostgreSQL user: $POSTGRES_USER"
        echo "PostgreSQL database: $POSTGRES_DB"
        echo "Trying to ping the host:"
        ping -c 3 $POSTGRES_HOST || echo "Unable to ping host"
        exit 1
    fi
    echo "Waiting for PostgreSQL to be available... ($timeout seconds left)"
    sleep 1
done

echo "PostgreSQL is now available"

# Run the application
exec python3 src/main.py 