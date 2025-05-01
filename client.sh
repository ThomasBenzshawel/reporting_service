#!/bin/bash
# Analytics Service Client Script
# This script provides easy access to the analytics service endpoints

# Set default values
SERVICE_URL=${ANALYTICS_URL:-"http://localhost:8000"}
USERNAME=""
PASSWORD=""
OUTPUT_DIR="./analytics_output"

# Function to print usage
function print_usage {
  echo "Usage: $0 [options] <endpoint>"
  echo ""
  echo "Options:"
  echo "  -u, --username EMAIL    Your email for authentication"
  echo "  -p, --password PWD      Your password"
  echo "  -s, --server URL        Analytics server URL (default: $SERVICE_URL)"
  echo "  -o, --output DIR        Output directory (default: $OUTPUT_DIR)"
  echo "  -h, --help              Show this help message"
  echo ""
  echo "Available endpoints:"
  echo "  ratings                 Histogram of accuracy/completeness ratings"
  echo "  time                    Distribution of time spent on evaluations"
  echo "  unknown                 Analysis of unknown object markings"
  echo "  hallucination           Analysis of hallucination markings"
  echo "  disagreement            Rating disagreement analysis"
  echo "  users-csv               Export users statistics to CSV"
  echo "  objects-csv             Export objects statistics to CSV"
  echo "  dashboard               Open the analytics dashboard in browser"
  echo ""
  echo "Example:"
  echo "  $0 -u admin@example.com -p password123 ratings"
}

# Check for required tools
for cmd in curl jq; do
  if ! command -v $cmd &> /dev/null; then
    echo "Error: $cmd is required but not installed. Please install it and try again."
    exit 1
  fi
done

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    -u|--username)
      USERNAME="$2"
      shift
      shift
      ;;
    -p|--password)
      PASSWORD="$2"
      shift
      shift
      ;;
    -s|--server)
      SERVICE_URL="$2"
      shift
      shift
      ;;
    -o|--output)
      OUTPUT_DIR="$2"
      shift
      shift
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    *)
      ENDPOINT="$1"
      shift
      ;;
  esac
done

# Check if endpoint is provided
if [ -z "$ENDPOINT" ]; then
  echo "Error: No endpoint specified"
  print_usage
  exit 1
fi

# Check if credentials are provided
if [ -z "$USERNAME" ] || [ -z "$PASSWORD" ]; then
  echo "Error: Username and password are required"
  print_usage
  exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Map endpoint to API path and filename
case $ENDPOINT in
  ratings)
    API_PATH="/analytics/histogram/ratings"
    OUTPUT_FILE="$OUTPUT_DIR/ratings_histogram.png"
    ;;
  time)
    API_PATH="/analytics/distribution/time_spent"
    OUTPUT_FILE="$OUTPUT_DIR/time_spent.png"
    ;;
  unknown)
    API_PATH="/analytics/unknown_count"
    OUTPUT_FILE="$OUTPUT_DIR/unknown_count.png"
    ;;
  hallucination)
    API_PATH="/analytics/hallucination_count"
    OUTPUT_FILE="$OUTPUT_DIR/hallucination_count.png"
    ;;
  disagreement)
    API_PATH="/analytics/rating_disagreement"
    OUTPUT_FILE="$OUTPUT_DIR/rating_disagreement.png"
    ;;
  users-csv)
    API_PATH="/analytics/export/users_csv"
    OUTPUT_FILE="$OUTPUT_DIR/users_with_stats.csv"
    ;;
  objects-csv)
    API_PATH="/analytics/export/objects_csv"
    OUTPUT_FILE="$OUTPUT_DIR/objects_with_stats.csv"
    ;;
  dashboard)
    # For dashboard, we'll just open it in the browser
    echo "Opening dashboard in browser..."
    if command -v xdg-open &> /dev/null; then
      xdg-open "$SERVICE_URL/analytics/dashboard"
    elif command -v open &> /dev/null; then
      open "$SERVICE_URL/analytics/dashboard"
    else
      echo "Cannot open browser automatically. Please visit:"
      echo "$SERVICE_URL/analytics/dashboard"
    fi
    exit 0
    ;;
  *)
    echo "Error: Unknown endpoint '$ENDPOINT'"
    print_usage
    exit 1
    ;;
esac

# Get token
echo "Authenticating..."
TOKEN=$(curl -s -X POST "$SERVICE_URL/token" \
  -d "username=$USERNAME&password=$PASSWORD" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  | jq -r '.access_token')

# Check if token was obtained
if [ -z "$TOKEN" ] || [ "$TOKEN" == "null" ]; then
  echo "Error: Failed to authenticate. Please check your credentials."
  exit 1
fi

echo "Authentication successful!"

# Make the API request
echo "Requesting $ENDPOINT data..."
HTTP_CODE=$(curl -s -o "$OUTPUT_FILE" -w "%{http_code}" \
  -X GET "$SERVICE_URL$API_PATH" \
  -H "Authorization: Bearer $TOKEN")

# Check response
if [ "$HTTP_CODE" -eq 200 ]; then
  echo "Success! Data saved to: $OUTPUT_FILE"
  
  # For images, try to display them
  if [[ "$OUTPUT_FILE" == *.png ]]; then
    if command -v display &> /dev/null; then
      display "$OUTPUT_FILE" &
    elif command -v open &> /dev/null; then
      open "$OUTPUT_FILE"
    elif command -v xdg-open &> /dev/null; then
      xdg-open "$OUTPUT_FILE"
    else
      echo "Image viewer not found. Please open the file manually."
    fi
  # For CSVs, show a preview
  elif [[ "$OUTPUT_FILE" == *.csv ]]; then
    echo "CSV Preview (first 5 rows):"
    head -n 5 "$OUTPUT_FILE"
  fi
else
  echo "Error: Request failed with HTTP code $HTTP_CODE"
  cat "$OUTPUT_FILE"  # Show error message
  rm "$OUTPUT_FILE"   # Remove error file
  exit 1
fi