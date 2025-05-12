from fastapi import FastAPI, Depends, Request, Header, HTTPException, Form, status
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse, JSONResponse, RedirectResponse
import httpx
import io
import csv
import os
import json
import logging
from typing import Optional, List, Dict, Any
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import base64
from io import BytesIO
import pandas as pd
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from starlette.middleware.sessions import SessionMiddleware
from dotenv import load_dotenv

load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("analytics-service")

# Configuration
API_URL = os.getenv("API_URL", "http://api-service:8000")
AUTH_URL = os.getenv("AUTH_URL", "http://auth-service:8000")
SESSION_SECRET = os.getenv("SESSION_SECRET_KEY", "supersecretkey-change-in-production")

# Create app
app = FastAPI(title="Analytics Service")

# Add session middleware
app.add_middleware(
    SessionMiddleware, 
    secret_key=SESSION_SECRET,
    max_age=3600  # 1 hour expiry
)

# Define paths for templates and static files
base_dir = os.path.dirname(os.path.abspath(__file__))
templates_dir = os.path.join(base_dir, "templates")
static_dir = os.path.join(base_dir, "static")

# Set up static files and templates
app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=templates_dir)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Pydantic Models
class User(BaseModel):
    userId: str
    email: str
    role: str

class Token(BaseModel):
    access_token: str
    token_type: str

# Auth helper function to get token
async def get_token(username: str, password: str) -> Optional[str]:
    """Get an authentication token from the auth service"""
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.post(
                f"{AUTH_URL}/login",
                data={"username": username, "password": password}
            )
            
            if response.status_code == 200:
                return response.json().get("access_token")
            logger.warning(f"Failed to get token: {response.status_code} - {response.text}")
            return None
        except Exception as e:
            logger.error(f"Error getting token: {e}")
            return None

# Auth dependency that supports both session tokens and authorization headers (since we do both cli and web app)
async def get_current_user(request: Request, authorization: Optional[str] = Header(None)):
    # Get token from session
    token = request.session.get("access_token")

    # If no token in session, try from header
    if not token and authorization and authorization.startswith("Bearer "):
        token = authorization.replace("Bearer ", "")    

    if not token:
        logger.info("No token found in session or header")
        # Return None instead of raising an exception
        return None
    
    try:
        # Get user info from auth service
        async with httpx.AsyncClient(timeout=10.0) as client:
            headers = {"Authorization": f"Bearer {token}"}
            response = await client.get(f"{AUTH_URL}/me", headers=headers)
            if response.status_code != 200:
                logger.warning(f"Failed to verify token: {response.status_code}")
                return None
            user_data = response.json()
            return User(**user_data)
    except Exception as e:
        logger.error(f"Error verifying user: {str(e)}")
        return None

# Admin access check
async def check_admin_access(user: Optional[User] = Depends(get_current_user)):
    """Verify the user is authenticated and has admin role"""
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    return user

# helper function to fetch all objects with pagination
async def fetch_all_objects(request: Request, authorization: Optional[str] = Header(None)) -> List[Dict[str, Any]]:
    """Fetch all objects from the API with pagination"""
    all_objects = []
    page = 1
    limit = 100
    
    # Get token from header or session
    token = None
    if authorization and isinstance(authorization, str) and authorization.startswith("Bearer "):
        token = authorization.replace("Bearer ", "")    
    if not token:
        token = request.session.get("access_token")
    
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        headers = {"Authorization": f"Bearer {token}"}
        
        while True:
            try:
                response = await client.get(f"{API_URL}/api/objects?page={page}&limit={limit}", headers=headers)
                
                if response.status_code != 200:
                    logger.error(f"Failed to fetch objects: {response.status_code} - {response.text}")
                    raise HTTPException(
                        status_code=response.status_code, 
                        detail=f"Failed to fetch objects: {response.text}"
                    )
                
                data = response.json()
                objects = data.get("data", [])
                all_objects.extend(objects)
                
                # Check if we've reached the last page
                if page >= data.get("pages", 0) or len(objects) == 0:
                    break
                
                page += 1
            except httpx.TimeoutException:
                logger.error(f"Timeout when fetching objects page {page}")
                raise HTTPException(
                    status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                    detail="Timed out while fetching objects from API"
                )
            except Exception as e:
                logger.error(f"Error fetching objects: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                    detail=f"Error fetching objects: {str(e)}"
                )
    
    return all_objects

# Utility function to create and save plots
def create_plot(plt_func, filename, *args, **kwargs):
    """Create a plot and save it to a BytesIO object"""
    plt.figure(figsize=(10, 6))
    plt_func(*args, **kwargs)
    plt.tight_layout()
    
    # Save plot to BytesIO object
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    
    return buf

# Error handling wrapper for visualization endpoints
def handle_visualization_errors(func):
    async def wrapper(request: Request, *args, **kwargs):
        try:
            # Get admin user manually
            token = request.session.get("access_token")
            if not token:
                # Return error image for unauthorized requests
                return create_error_image("Authentication required")
                
            # Manually verify admin access
            async with httpx.AsyncClient(timeout=10.0) as client:
                headers = {"Authorization": f"Bearer {token}"}
                response = await client.get(f"{AUTH_URL}/me", headers=headers)
                if response.status_code != 200:
                    return create_error_image("Invalid authentication")
                
                user_data = response.json()
                user = User(**user_data)
                
                if user.role != "admin":
                    return create_error_image("Admin access required")
                
            # Call the original function with the admin user
            return await func(request, admin_user=user, *args, **kwargs)
        except Exception as e:
            logger.error(f"Error generating visualization: {e}")
            return create_error_image(f"Error: {str(e)}")
    return wrapper

# Create a helper function for error images
def create_error_image(error_text):
    plt.figure(figsize=(10, 6))
    plt.text(0.5, 0.5, error_text, ha='center', va='center', transform=plt.gca().transAxes)
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return StreamingResponse(buf, media_type="image/png")

# routes
@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, user: Optional[User] = Depends(get_current_user)):
    if user:
        return RedirectResponse(url="/analytics/dashboard")
    
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login")
async def login(request: Request, email: str = Form(...), password: str = Form(...)):
    # Call auth service to get token
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{AUTH_URL}/login",
                data={"username": email, "password": password}
            )
            
            if response.status_code != 200:
                logger.warning(f"Auth service login error: {response.status_code} - {response.text}")
                return templates.TemplateResponse(
                    "login.html",
                    {"request": request, "error": f"Login failed: {response.status_code} - {response.text}"}
                )
            
            token_data = response.json()
            logger.info(f"Got token with type: {type(token_data.get('access_token'))}")
            
            # Store token in session
            request.session["access_token"] = token_data["access_token"]
            logger.info(f"Stored token in session: {request.session.get('access_token') is not None}")
            
            return RedirectResponse(url="/analytics/dashboard", status_code=303)
    except Exception as e:
        logger.error(f"Exception during login: {str(e)}")
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error": str(e)}
        )

@app.get("/logout")
async def logout(request: Request):
    # Remove token from session
    request.session.pop("access_token", None)
    
    return RedirectResponse(url="/login")

# does the visualization endpoints
async def get_viz_admin_user(request: Request):
    """Special authentication for visualization endpoints that bypasses dependency validation"""
    # Get token from session
    token = request.session.get("access_token")
    
    if not token:
        return None
    
    try:
        # Verify admin access
        async with httpx.AsyncClient(timeout=10.0) as client:
            headers = {"Authorization": f"Bearer {token}"}
            response = await client.get(f"{AUTH_URL}/me", headers=headers)
            if response.status_code != 200:
                logger.warning(f"Invalid token for visualization: {response.status_code}")
                return None
            
            user_data = response.json()
            user = User(**user_data)
            
            if user.role != "admin":
                logger.warning(f"Non-admin user attempted to access visualization: {user.email}")
                return None
            
            return user
    except Exception as e:
        logger.error(f"Error verifying user for visualization: {str(e)}")
        return None

# if a service is down, create an error image
def create_error_image(error_message):
    """Create an error image with the specified message"""
    plt.figure(figsize=(10, 6))
    plt.text(0.5, 0.5, error_message, ha='center', va='center', transform=plt.gca().transAxes)
    plt.tight_layout()
    
    # Save plot to BytesIO object
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    
    return buf

# Histogram ratings endpoint
@app.get("/analytics/histogram/ratings")
async def histogram_ratings(request: Request):
    """Generate histograms for accuracy and completeness ratings"""
    # Get admin user manually without dependency
    admin_user = await get_viz_admin_user(request)
    
    if not admin_user:
        # Return error image instead of raising an exception
        error_buf = create_error_image("Authentication required")
        return StreamingResponse(error_buf, media_type="image/png")
    
    try:
        # Original visualization code
        all_objects = await fetch_all_objects(request)
        
        # Extract accuracy and completeness ratings
        accuracy_ratings = []
        completeness_ratings = []
        
        for obj in all_objects:
            avg_ratings = obj.get("averageRatings", {})
            if "accuracy" in avg_ratings and avg_ratings["accuracy"] is not None:
                accuracy_ratings.append(avg_ratings["accuracy"])
            if "completeness" in avg_ratings and avg_ratings["completeness"] is not None:
                completeness_ratings.append(avg_ratings["completeness"])
        
        # Handle empty data case
        if not accuracy_ratings and not completeness_ratings:
            plt.figure(figsize=(15, 6))
            plt.text(0.5, 0.5, "No rating data available", ha='center', va='center', transform=plt.gca().transAxes)
            plt.tight_layout()
            buf = BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plt.close()
            return StreamingResponse(buf, media_type="image/png")
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy histogram
        if accuracy_ratings:
            ax1.hist(accuracy_ratings, bins=10, alpha=0.7, color='blue')
            ax1.set_title('Distribution of Accuracy Ratings')
            ax1.set_xlabel('Accuracy Rating')
            ax1.set_ylabel('Count')
        else:
            ax1.text(0.5, 0.5, "No accuracy data available", ha='center', va='center', transform=ax1.transAxes)
        
        # Completeness histogram
        if completeness_ratings:
            ax2.hist(completeness_ratings, bins=10, alpha=0.7, color='green')
            ax2.set_title('Distribution of Completeness Ratings')
            ax2.set_xlabel('Completeness Rating')
            ax2.set_ylabel('Count')
        else:
            ax2.text(0.5, 0.5, "No completeness data available", ha='center', va='center', transform=ax2.transAxes)
        
        plt.tight_layout()
        
        # Save plot to BytesIO object
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        
        # Return image
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        logger.error(f"Error generating histogram: {e}")
        error_buf = create_error_image(f"Error generating visualization: {str(e)}")
        return StreamingResponse(error_buf, media_type="image/png")

# Time spent distribution endpoint
@app.get("/analytics/distribution/time_spent")
async def distribution_time_spent(request: Request):
    """Generate distribution of time spent on evaluations"""
    # Get admin user manually without dependency
    admin_user = await get_viz_admin_user(request)
    
    if not admin_user:
        # Return error image instead of raising an exception
        error_buf = create_error_image("Authentication required")
        return StreamingResponse(error_buf, media_type="image/png")
    
    try:
        # Original visualization code
        all_objects = await fetch_all_objects(request)
        
        # Extract time spent data
        time_spent_values = []
        
        for obj in all_objects:
            for rating in obj.get("ratings", []):
                time_spent = rating.get("metrics", {}).get("time_spent_seconds")
                if time_spent is not None:
                    time_spent_values.append(time_spent)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        
        if time_spent_values:
            plt.hist(time_spent_values, bins=20, alpha=0.7, color='purple')
            plt.title('Distribution of Time Spent on Evaluations')
            plt.xlabel('Time Spent (seconds)')
            plt.ylabel('Count')
            plt.grid(True, alpha=0.3)
            
            # Add statistics annotations
            avg_time = np.mean(time_spent_values)
            median_time = np.median(time_spent_values)
            plt.axvline(avg_time, color='red', linestyle='dashed', linewidth=1)
            plt.axvline(median_time, color='green', linestyle='dashed', linewidth=1)
            plt.text(avg_time*1.05, plt.ylim()[1]*0.9, f'Mean: {avg_time:.1f}s', color='red')
            plt.text(median_time*1.05, plt.ylim()[1]*0.8, f'Median: {median_time:.1f}s', color='green')
        else:
            plt.text(0.5, 0.5, "No time spent data available", ha='center', va='center', transform=plt.gca().transAxes)
        
        plt.tight_layout()
        
        # Save plot to BytesIO object
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        
        # Return image
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        logger.error(f"Error generating time spent distribution: {e}")
        error_buf = create_error_image(f"Error generating visualization: {str(e)}")
        return StreamingResponse(error_buf, media_type="image/png")

# Unknown count endpoint
@app.get("/analytics/unknown_count")
async def unknown_count(request: Request):
    """Generate visualization of unknown counts across objects"""
    # Get admin user manually without dependency
    admin_user = await get_viz_admin_user(request)
    
    if not admin_user:
        # Return error image instead of raising an exception
        error_buf = create_error_image("Authentication required")
        return StreamingResponse(error_buf, media_type="image/png")
    
    try:
        # Original visualization code
        all_objects = await fetch_all_objects(request)
        
        # Count objects with unknown markings
        object_unknown_counts = {}
        total_evaluations = 0
        total_unknown = 0
        
        for obj in all_objects:
            obj_id = obj.get("objectId", "unknown")
            unknown_count = 0
            eval_count = len(obj.get("ratings", []))
            
            for rating in obj.get("ratings", []):
                if rating.get("metrics", {}).get("unknown_object", False):
                    unknown_count += 1
                    total_unknown += 1
            
            total_evaluations += eval_count
            
            if unknown_count > 0:
                object_unknown_counts[obj_id] = {
                    "unknown_count": unknown_count,
                    "total_ratings": eval_count,
                    "percentage": (unknown_count / eval_count * 100) if eval_count > 0 else 0
                }
        
        # Sort by unknown count
        sorted_objects = sorted(
            object_unknown_counts.items(),
            key=lambda x: x[1]["unknown_count"],
            reverse=True
        )[:20]  # Top 20 objects
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        if sorted_objects:
            objects = [item[0][-6:] + "..." for item in sorted_objects]  # Truncate long object IDs
            unknown_counts = [item[1]["unknown_count"] for item in sorted_objects]
            percentages = [item[1]["percentage"] for item in sorted_objects]
            
            # Bar chart for counts
            bars = ax.bar(objects, unknown_counts, color='orangered')
            ax.set_xlabel('Object ID (truncated)')
            ax.set_ylabel('Number of Unknown Markings', color='orangered')
            ax.tick_params(axis='y', labelcolor='orangered')
            
            # Add percentage labels
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{percentages[i]:.1f}%',
                        ha='center', va='bottom', rotation=0, size=8)
            
            # Overall statistics
            overall_percentage = (total_unknown / total_evaluations * 100) if total_evaluations > 0 else 0
            plt.title(f'Objects with Most Unknown Markings\nTotal Unknown: {total_unknown} ({overall_percentage:.1f}% of all evaluations)')
            plt.xticks(rotation=45, ha='right')
        else:
            plt.text(0.5, 0.5, "No unknown objects found", ha='center', va='center', transform=ax.transAxes)
            plt.title('Unknown Objects Analysis')
        
        plt.tight_layout()
        
        # Save plot to BytesIO object
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        
        # Return image
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        logger.error(f"Error generating unknown count: {e}")
        error_buf = create_error_image(f"Error generating visualization: {str(e)}")
        return StreamingResponse(error_buf, media_type="image/png")

# Hallucination count endpoint
@app.get("/analytics/hallucination_count")
async def hallucination_count(request: Request):
    """Generate visualization of hallucination counts across objects"""
    # Get admin user manually without dependency
    admin_user = await get_viz_admin_user(request)
    
    if not admin_user:
        # Return error image instead of raising an exception
        error_buf = create_error_image("Authentication required")
        return StreamingResponse(error_buf, media_type="image/png")
    
    try:
        # Original visualization code
        all_objects = await fetch_all_objects(request)
        
        # Count objects with hallucination markings
        object_hallucination_counts = {}
        total_evaluations = 0
        total_hallucinations = 0
        
        for obj in all_objects:
            obj_id = obj.get("objectId", "unknown")
            hallucination_count = 0
            eval_count = len(obj.get("ratings", []))
            
            for rating in obj.get("ratings", []):
                if rating.get("metrics", {}).get("hallucinated", False):
                    hallucination_count += 1
                    total_hallucinations += 1
            
            total_evaluations += eval_count
            
            if hallucination_count > 0:
                object_hallucination_counts[obj_id] = {
                    "hallucination_count": hallucination_count,
                    "total_ratings": eval_count,
                    "percentage": (hallucination_count / eval_count * 100) if eval_count > 0 else 0
                }
        
        # Sort by hallucination count
        sorted_objects = sorted(
            object_hallucination_counts.items(),
            key=lambda x: x[1]["hallucination_count"],
            reverse=True
        )[:20]  # Top 20 objects
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        if sorted_objects:
            objects = [item[0][-6:] + "..." for item in sorted_objects]  # Truncate long object IDs
            hallucination_counts = [item[1]["hallucination_count"] for item in sorted_objects]
            percentages = [item[1]["percentage"] for item in sorted_objects]
            
            # Bar chart for counts
            bars = ax.bar(objects, hallucination_counts, color='magenta')
            ax.set_xlabel('Object ID (truncated)')
            ax.set_ylabel('Number of Hallucination Markings', color='magenta')
            ax.tick_params(axis='y', labelcolor='magenta')
            
            # Add percentage labels
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{percentages[i]:.1f}%',
                        ha='center', va='bottom', rotation=0, size=8)
            
            # Overall statistics
            overall_percentage = (total_hallucinations / total_evaluations * 100) if total_evaluations > 0 else 0
            plt.title(f'Objects with Most Hallucination Markings\nTotal Hallucinations: {total_hallucinations} ({overall_percentage:.1f}% of all evaluations)')
            plt.xticks(rotation=45, ha='right')
        else:
            plt.text(0.5, 0.5, "No hallucinations found", ha='center', va='center', transform=ax.transAxes)
            plt.title('Hallucinations Analysis')
        
        plt.tight_layout()
        
        # Save plot to BytesIO object
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        
        # Return image
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        logger.error(f"Error generating hallucination count: {e}")
        error_buf = create_error_image(f"Error generating visualization: {str(e)}")
        return StreamingResponse(error_buf, media_type="image/png")

# Updated rating disagreement endpoint
@app.get("/analytics/rating_disagreement")
async def rating_disagreement(request: Request):
    """Analyze rating disagreement for objects with multiple ratings"""
    # Get admin user manually without dependency
    admin_user = await get_viz_admin_user(request)
    
    if not admin_user:
        # Return error image instead of raising an exception
        error_buf = create_error_image("Authentication required")
        return StreamingResponse(error_buf, media_type="image/png")
    
    try:
        # Original visualization code
        all_objects = await fetch_all_objects(request)
        
        # Filter for objects with multiple ratings
        multi_rated_objects = [obj for obj in all_objects if len(obj.get("ratings", [])) > 1]
        
        # Calculate disagreement metrics
        accuracy_diffs = []
        completeness_diffs = []
        accuracy_std_devs = []
        completeness_std_devs = []
        
        # Count objects with disagreement
        accuracy_disagreement_count = 0
        completeness_disagreement_count = 0
        total_multi_rated = len(multi_rated_objects)
        
        # Disagreement threshold (difference of 2 or more points is considered disagreement)
        threshold = 2
        
        for obj in multi_rated_objects:
            # Extract all ratings for this object
            accuracy_ratings = []
            completeness_ratings = []
            
            for rating in obj.get("ratings", []):
                metrics = rating.get("metrics", {})
                if "accuracy" in metrics and metrics["accuracy"] is not None:
                    accuracy_ratings.append(metrics["accuracy"])
                if "completeness" in metrics and metrics["completeness"] is not None:
                    completeness_ratings.append(metrics["completeness"])
            
            # Calculate max difference and standard deviation for each metric
            if len(accuracy_ratings) > 1:
                max_acc_diff = max(accuracy_ratings) - min(accuracy_ratings)
                accuracy_diffs.append(max_acc_diff)
                accuracy_std_devs.append(np.std(accuracy_ratings))
                if max_acc_diff >= threshold:
                    accuracy_disagreement_count += 1
            
            if len(completeness_ratings) > 1:
                max_comp_diff = max(completeness_ratings) - min(completeness_ratings)
                completeness_diffs.append(max_comp_diff)
                completeness_std_devs.append(np.std(completeness_ratings))
                if max_comp_diff >= threshold:
                    completeness_disagreement_count += 1
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(15, 10))
        
        # Handle case with no data
        if not multi_rated_objects:
            plt.text(0.5, 0.5, "No objects with multiple ratings found", 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.tight_layout()
            buf = BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plt.close()
            return StreamingResponse(buf, media_type="image/png")
        
        # 1. Disagreement rates
        ax1 = fig.add_subplot(221)
        labels = ['Accuracy', 'Completeness']
        if total_multi_rated > 0:
            rates = [
                accuracy_disagreement_count / total_multi_rated * 100,
                completeness_disagreement_count / total_multi_rated * 100
            ]
            ax1.bar(labels, rates, color=['blue', 'green'])
            ax1.set_ylabel('Disagreement Rate (%)')
            ax1.set_title(f'Rating Disagreement Rates\n(Threshold: {threshold} points difference)')
            
            # Add percentage labels
            for i, rate in enumerate(rates):
                ax1.text(i, rate + 1, f'{rate:.1f}%', ha='center')
        else:
            ax1.text(0.5, 0.5, "No multi-rated objects found", ha='center', va='center', transform=ax1.transAxes)
        
        # 2. Histogram of accuracy differences
        ax2 = fig.add_subplot(222)
        if accuracy_diffs:
            bins = np.arange(0, max(accuracy_diffs) + 1.5) - 0.5
            if len(bins) <= 1:  # Handle edge case with single value
                bins = 3
            ax2.hist(accuracy_diffs, bins=bins, alpha=0.7, color='blue')
            ax2.set_title('Histogram of Max Accuracy Differences')
            ax2.set_xlabel('Max Difference Between Ratings')
            ax2.set_ylabel('Count')
            ax2.axvline(x=threshold-0.5, color='red', linestyle='--', label=f'Threshold ({threshold})')
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, "No accuracy data available", ha='center', va='center', transform=ax2.transAxes)
        
        # 3. Histogram of completeness differences
        ax3 = fig.add_subplot(223)
        if completeness_diffs:
            bins = np.arange(0, max(completeness_diffs) + 1.5) - 0.5
            if len(bins) <= 1:  # Handle edge case with single value
                bins = 3
            ax3.hist(completeness_diffs, bins=bins, alpha=0.7, color='green')
            ax3.set_title('Histogram of Max Completeness Differences')
            ax3.set_xlabel('Max Difference Between Ratings')
            ax3.set_ylabel('Count')
            ax3.axvline(x=threshold-0.5, color='red', linestyle='--', label=f'Threshold ({threshold})')
            ax3.legend()
        else:
            ax3.text(0.5, 0.5, "No completeness data available", ha='center', va='center', transform=ax3.transAxes)
        
        # 4. Standard deviation comparison
        ax4 = fig.add_subplot(224)
        if accuracy_std_devs and completeness_std_devs:
            data = [accuracy_std_devs, completeness_std_devs]
            ax4.boxplot(data, labels=['Accuracy', 'Completeness'])
            ax4.set_title('Standard Deviation of Ratings')
            ax4.set_ylabel('Standard Deviation')
            
            # Add mean annotations
            acc_mean = np.mean(accuracy_std_devs)
            comp_mean = np.mean(completeness_std_devs)
            ax4.text(1, acc_mean, f'Mean: {acc_mean:.2f}', ha='center', va='bottom')
            ax4.text(2, comp_mean, f'Mean: {comp_mean:.2f}', ha='center', va='bottom')
        else:
            ax4.text(0.5, 0.5, "Insufficient data for standard deviation analysis", ha='center', va='center', transform=ax4.transAxes)
        
        plt.tight_layout()
        
        # Save plot to BytesIO object
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        
        # Return image
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        logger.error(f"Error generating rating disagreement: {e}")
        error_buf = create_error_image(f"Error generating visualization: {str(e)}")
        return StreamingResponse(error_buf, media_type="image/png")
    

# Update analytics dashboard to redirect to login if not authenticated
@app.get("/analytics/dashboard", response_class=HTMLResponse)
async def analytics_dashboard(request: Request, user: Optional[User] = Depends(get_current_user)):
    """Dashboard displaying all analytics visualizations"""
    if not user:
        return RedirectResponse(url="/login")
    
    # For admin-only pages, check admin role
    if user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    # Render the dashboard template
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request, 
            "user_email": user.email,
            "title": "Analytics Dashboard - Objaverse Research Portal"
        }
    )

# CSV export endpoints
@app.get("/analytics/export/users_csv")
async def export_users_csv(request: Request, admin_user: User = Depends(check_admin_access)):
    """Export users data with evaluation statistics to CSV"""
   
    # Get all users
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            headers = {"Authorization": f"Bearer {request.session.get('access_token')}"}
            response = await client.get(f"{AUTH_URL}/users", headers=headers)
            if response.status_code != 200:
                logger.error(f"Failed to fetch users: {response.status_code} - {response.text}")
                raise HTTPException(status_code=404, detail="Users not found")
           
            users_data = response.json().get("data", [])
        except Exception as e:
            logger.error(f"Error fetching users: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error fetching users: {str(e)}"
            )
   
    # Create CSV with additional fields for unknown object statistics
    output = io.StringIO()
    fieldnames = ["userId", "email", "role", "total_evaluations", "unknown_objects", "unknown_percent", 
                 "hallucinated_objects", "hallucinated_percent", "avg_time_spent"]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
   
    # For each user, fetch their evaluations to calculate statistics
    for user_item in users_data:
        user_id = user_item.get("userId", "")
        
        # Get completed evaluations for this user
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                headers = {"Authorization": f"Bearer {request.session.get('access_token')}"}
                response = await client.post(
                    f"{API_URL}/api/completed",
                    json={"userId": user_id},
                    headers=headers
                )
                
                total_evaluations = 0
                unknown_count = 0
                hallucinated_count = 0
                total_time_spent = 0
                unknown_percent = 0
                hallucinated_percent = 0
                avg_time_spent = 0
                
                if response.status_code == 200:
                    completed_data = response.json().get("data", [])
                    total_evaluations = len(completed_data)
                    
                    # Count metrics
                    for obj in completed_data:
                        for rating in obj.get("ratings", []):
                            if rating.get("userId") == user_id:
                                metrics = rating.get("metrics", {})
                                if metrics.get("unknown_object", False):
                                    unknown_count += 1
                                if metrics.get("hallucinated", False):
                                    hallucinated_count += 1
                                if "time_spent_seconds" in metrics:
                                    total_time_spent += metrics["time_spent_seconds"]
                                break
                    
                    # Calculate percentages and averages
                    if total_evaluations > 0:
                        unknown_percent = round((unknown_count / total_evaluations) * 100, 1)
                        hallucinated_percent = round((hallucinated_count / total_evaluations) * 100, 1)
                        avg_time_spent = round(total_time_spent / total_evaluations, 1)
            except Exception as e:
                logger.error(f"Error fetching completed evaluations for user {user_id}: {e}")
                # Continue to next user
                continue
        
        # Write user data with stats to CSV
        writer.writerow({
            "userId": user_id,
            "email": user_item.get("email", ""),
            "role": user_item.get("role", ""),
            "total_evaluations": total_evaluations,
            "unknown_objects": unknown_count,
            "unknown_percent": f"{unknown_percent}%",
            "hallucinated_objects": hallucinated_count,
            "hallucinated_percent": f"{hallucinated_percent}%",
            "avg_time_spent": f"{avg_time_spent}s"
        })
   
    output.seek(0)
   
    # Return CSV file
    response = StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv"
    )
    response.headers["Content-Disposition"] = "attachment; filename=users_with_stats.csv"
   
    return response

@app.get("/analytics/export/objects_csv")
async def export_objects_csv(request: Request, admin_user: User = Depends(check_admin_access)):
    """Export objects data with evaluation statistics to CSV"""
   
    all_objects = await fetch_all_objects(request)
   
    # Create CSV with statistics fields
    output = io.StringIO()
    fieldnames = [
        "objectId", 
        "description", 
        "category", 
        "averageAccuracy", 
        "averageCompleteness", 
        "averageClarity", 
        "totalEvaluations", 
        "unknownCount", 
        "unknownPercent",
        "hallucinatedCount", 
        "hallucinatedPercent",
        "timeSpentAvg",
        "accuracyStdDev",
        "completenessStdDev",
        "maxAccuracyDiff",
        "maxCompletenessDiff",
        "createdAt"
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
   
    for obj in all_objects:
        # Calculate statistics
        total_evaluations = len(obj.get("ratings", []))
        unknown_count = 0
        hallucinated_count = 0
        total_time_spent = 0
        
        # Lists for calculating standard deviation and max differences
        accuracy_ratings = []
        completeness_ratings = []
        
        for rating in obj.get("ratings", []):
            metrics = rating.get("metrics", {})
            if metrics.get("unknown_object", False):
                unknown_count += 1
            if metrics.get("hallucinated", False):
                hallucinated_count += 1
            if "time_spent_seconds" in metrics:
                total_time_spent += metrics["time_spent_seconds"]
            
            # Collect ratings for std dev calculation
            if "accuracy" in metrics and metrics["accuracy"] is not None:
                accuracy_ratings.append(metrics["accuracy"])
            if "completeness" in metrics and metrics["completeness"] is not None:
                completeness_ratings.append(metrics["completeness"])
        
        # Calculate percentages and averages
        unknown_percent = 0
        hallucinated_percent = 0
        time_spent_avg = 0
        accuracy_std_dev = 0
        completeness_std_dev = 0
        max_accuracy_diff = 0
        max_completeness_diff = 0
        
        if total_evaluations > 0:
            unknown_percent = round((unknown_count / total_evaluations) * 100, 1)
            hallucinated_percent = round((hallucinated_count / total_evaluations) * 100, 1)
            time_spent_avg = round(total_time_spent / total_evaluations, 1)
        
        # Calculate std dev and max differences if multiple ratings exist
        if len(accuracy_ratings) > 1:
            accuracy_std_dev = round(np.std(accuracy_ratings), 2)
            max_accuracy_diff = max(accuracy_ratings) - min(accuracy_ratings)
        
        if len(completeness_ratings) > 1:
            completeness_std_dev = round(np.std(completeness_ratings), 2)
            max_completeness_diff = max(completeness_ratings) - min(completeness_ratings)

        average_ratings = obj.get("averageRatings", {})
        
        # Handle potential None values safely
        description = obj.get("description", "")
        if description is None:
            description = ""
            
        writer.writerow({
            "objectId": obj.get("objectId", ""),
            "description": description[:100] + "..." if len(description) > 100 else description,  # Truncate long descriptions
            "category": obj.get("category", ""),
            "averageAccuracy": average_ratings.get("accuracy", 0),
            "averageCompleteness": average_ratings.get("completeness", 0),
            "averageClarity": average_ratings.get("clarity", 0),
            "totalEvaluations": total_evaluations,
            "unknownCount": unknown_count,
            "unknownPercent": f"{unknown_percent}%",
            "hallucinatedCount": hallucinated_count,
            "hallucinatedPercent": f"{hallucinated_percent}%",
            "timeSpentAvg": f"{time_spent_avg}s",
            "accuracyStdDev": accuracy_std_dev,
            "completenessStdDev": completeness_std_dev,
            "maxAccuracyDiff": max_accuracy_diff,
            "maxCompletenessDiff": max_completeness_diff,
            "createdAt": obj.get("createdAt", "")
        })
   
    output.seek(0)
   
    # Return CSV file
    response = StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv"
    )
    response.headers["Content-Disposition"] = "attachment; filename=objects_with_detailed_stats.csv"
   
    return response

# Token endpoint for curl authentication (for the cli client)
@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Authenticate user and return JWT token.
    This endpoint is used for curl requests to get an authentication token.
    """
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            # Forward the request to the auth service
            response = await client.post(
                f"{AUTH_URL}/login",
                data={"username": form_data.username, "password": form_data.password}
            )
            
            if response.status_code != 200:
                logger.warning(f"Authentication failed: {response.status_code} - {response.text}")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid username or password",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            
            token_data = response.json()
            return token_data
        except httpx.TimeoutException:
            logger.error("Authentication timeout")
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Authentication service timeout",
            )
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Authentication error: {str(e)}",
            )

# Main index route that redirects to the dashboard
@app.get("/")
async def root(request: Request, user: Optional[User] = Depends(get_current_user)):
    """Redirect to dashboard page if authenticated, otherwise to login"""
    if not user:
        return RedirectResponse(url="/login")
    return RedirectResponse(url="/analytics/dashboard")

# Data endpoint to provide summary statistics
@app.get("/analytics/data")
async def analytics_data(request: Request, admin_user: User = Depends(check_admin_access)):
    """Provide summary statistics for the dashboard"""
    
    all_objects = await fetch_all_objects(request)
    
    # Calculate summary statistics
    total_objects = len(all_objects)
    total_evaluations = sum(len(obj.get("ratings", [])) for obj in all_objects)
    
    # Count unknown objects
    unknown_objects = 0
    hallucinations = 0
    
    for obj in all_objects:
        for rating in obj.get("ratings", []):
            if rating.get("metrics", {}).get("unknown_object", False):
                unknown_objects += 1
            if rating.get("metrics", {}).get("hallucinated", False):
                hallucinations += 1
    
    return {
        "total_objects": total_objects,
        "total_evaluations": total_evaluations,
        "unknown_objects": unknown_objects,
        "hallucinations": hallucinations
    }

# Endpoint for health checks for deployment
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "service": "analytics"}

@app.exception_handler(401)
async def unauthorized_handler(request: Request, exc: HTTPException):
    """Handle unauthorized access by redirecting to login"""
    return RedirectResponse(url="/login", status_code=303)

@app.exception_handler(403)
async def forbidden_handler(request: Request, exc: HTTPException):
    """Handle forbidden access"""
    return templates.TemplateResponse(
        "error.html",
        {
            "request": request,
            "status_code": 403,
            "status_message": "Forbidden",
            "detail": "You don't have permission to access this resource"
        },
        status_code=403
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info", workers=3)