"""
Formatting utilities for display.
"""
from typing import Union, Optional
import pandas as pd


def format_number(value: Union[int, float], decimals: int = 2, 
                 prefix: str = "", suffix: str = "") -> str:
    """
    Format a number for display with proper separators.
    
    Args:
        value: Number to format
        decimals: Number of decimal places
        prefix: Prefix to add (e.g., '$')
        suffix: Suffix to add (e.g., '%')
        
    Returns:
        Formatted string
    """
    if pd.isna(value):
        return "N/A"
    
    if decimals == 0:
        formatted = f"{value:,.0f}"
    else:
        formatted = f"{value:,.{decimals}f}"
    
    return f"{prefix}{formatted}{suffix}"


def format_percentage(value: Union[int, float], decimals: int = 2) -> str:
    """Format a value as a percentage."""
    return format_number(value, decimals, suffix="%")


def format_currency(value: Union[int, float], decimals: int = 2) -> str:
    """Format a value as currency."""
    return format_number(value, decimals, prefix="$")


def format_date(date: pd.Timestamp, fmt: str = "%Y-%m-%d") -> str:
    """Format a pandas timestamp for display."""
    return date.strftime(fmt)


def format_delta(value: Union[int, float], decimals: int = 2, 
                positive_color: str = "green", negative_color: str = "red") -> str:
    """
    Format a delta value with color based on sign.
    
    Args:
        value: Delta value
        decimals: Number of decimal places
        positive_color: Color for positive values
        negative_color: Color for negative values
        
    Returns:
        HTML formatted string with color
    """
    if pd.isna(value):
        return "N/A"
    
    if value >= 0:
        color = positive_color
        prefix = "+"
    else:
        color = negative_color
        prefix = ""
    
    formatted = f"{prefix}{value:.{decimals}f}%"
    return f'<span style="color: {color};">{formatted}</span>'


def format_ratio(value: Union[int, float], decimals: int = 2) -> str:
    """Format a ratio value."""
    return format_number(value, decimals)


def get_period_label(start_date: pd.Timestamp, end_date: pd.Timestamp) -> str:
    """
    Get a formatted label for the date period.
    
    Args:
        start_date: Start date
        end_date: End date
        
    Returns:
        Formatted period string
    """
    days = (end_date - start_date).days
    years = days / 365.25
    
    if years >= 1:
        return f"{format_date(start_date)} to {format_date(end_date)} ({years:.1f} years)"
    else:
        return f"{format_date(start_date)} to {format_date(end_date)} ({days} days)"
