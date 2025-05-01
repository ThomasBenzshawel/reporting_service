# Evaluation Analytics Service

This service provides analytics and visualization capabilities for the objaverse evaluation system.

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

This service requires authentication. Only Admins can access the analytics endpoints.

### Making Requests

You can access visualizations directly via curl:

```bash
# Get histogram of accuracy and completeness ratings
curl -X GET "http://analytics-service:8000/analytics/histogram/ratings" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -o ratings_histogram.png

# Get time spent distribution
curl -X GET "http://analytics-service:8000/analytics/distribution/time_spent" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -o time_spent.png

# Export users data to CSV
curl -X GET "http://analytics-service:8000/analytics/export/users_csv" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -o users_with_stats.csv
```

### Dashboard Access

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
docker run -p 8000:8000 \
  -e API_URL=http://your-api-service:8000 \
  -e AUTH_URL=http://your-auth-service:8000 \
  evaluation-analytics
```

### Using Docker Compose

You can add this service to your docker-compose.yml file:

```yaml
services:
  analytics:
    build: ./analytics
    ports:
      - "8000:8000"
    environment:
      - API_URL=http://api:8000
      - AUTH_URL=http://auth:8000
    depends_on:
      - api
      - auth
```
