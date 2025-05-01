# Evaluation Analytics Service

This service provides analytics and visualizations for the objaverse validation service. 

## Endpoints

### Visualization Endpoints

- **GET /analytics/histogram/ratings**: Histogram of accuracy and completeness across all objects
- **GET /analytics/distribution/time_spent**: Distribution of time spent evaluating
- **GET /analytics/unknown_count**: Unknown count across all objects
- **GET /analytics/hallucination_count**: Hallucination count across all objects
- **GET /analytics/rating_disagreement**: Disagreement analysis for objects with multiple ratings

### Dashboard

- **GET /analytics/dashboard**: HTML dashboard displaying all analytics visualizations

### Export Endpoints

- **GET /analytics/export/users_csv**: Export user statistics to CSV
- **GET /analytics/export/objects_csv**: Export object statistics to CSV

## Using the Service

### Authentication

This service requires authentication. It reuses the same authentication system as your main application. Only users with the "admin" role can access the analytics endpoints.

### Authentication

The service supports two authentication methods:

1. **Web Interface**: When accessing through a browser, you'll use session-based authentication
2. **API Access**: When using curl or other API clients, you can use username/password to get a token

#### Getting an Authentication Token

To get a token for API access, use the `/token` endpoint:

```bash
# Get authentication token
TOKEN=$(curl -s -X POST "http://analytics-service:8000/token" \
  -d "username=your_email@example.com&password=your_password" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  | jq -r '.access_token')

echo "Token: $TOKEN"
```

### Making Requests

Once you have a token, you can access visualizations directly via curl:

```bash
# Get histogram of accuracy and completeness ratings
curl -X GET "http://analytics-service:8000/analytics/histogram/ratings" \
  -H "Authorization: Bearer $TOKEN" \
  -o ratings_histogram.png

# Get time spent distribution
curl -X GET "http://analytics-service:8000/analytics/distribution/time_spent" \
  -H "Authorization: Bearer $TOKEN" \
  -o time_spent.png

# Export users data to CSV
curl -X GET "http://analytics-service:8000/analytics/export/users_csv" \
  -H "Authorization: Bearer $TOKEN" \
  -o users_with_stats.csv
```

#### One-liner with Authentication

You can also use a one-line command that handles authentication and the request:

```bash
# Get histogram with inline authentication
curl -X GET "http://analytics-service:8000/analytics/histogram/ratings" \
  -H "Authorization: Bearer $(curl -s -X POST "http://analytics-service:8000/token" \
  -d "username=your_email@example.com&password=your_password" \
  -H "Content-Type: application/x-www-form-urlencoded" | jq -r '.access_token')" \
  -o ratings_histogram.png
```

Access the dashboard by navigating to:

```
http://analytics-service:8000/analytics/dashboard
```
## Deployment

### Environment Variables

Set the following environment variables:

- `API_URL`: URL of your main API service (default: "http://api-service:8000")
- `AUTH_URL`: URL of your auth service (default: "http://auth-service:8000")

### Using Docker

Build and run the service using Docker:

```bash
# Build the Docker image
docker build -t evaluation-analytics .

# Run the container
docker run -p 8000:8000 evaluation-analytics
```