#!/usr/bin/env python3
"""
Generate AIDI period list for GitHub Actions matrix
Outputs JSON array of {start, end, name} objects
"""
import json
import sys
from datetime import datetime, timedelta

START = sys.argv[1] if len(sys.argv) > 1 else '2022-12-01'
END = sys.argv[2] if len(sys.argv) > 2 else '2026-06-16'

start = datetime.strptime(START, '%Y-%m-%d')
end = datetime.strptime(END, '%Y-%m-%d')

periods = []
current = start.replace(day=1)

while current <= end:
    for day in [1, 16]:
        d = current.replace(day=min(day, 28))
        if d < start:
            continue
        if d > end:
            break
        
        # Calculate end date: day before next period
        if day == 1:
            next_d = current.replace(day=16)
        else:
            if current.month == 12:
                next_d = current.replace(year=current.year + 1, month=1, day=1)
            else:
                next_d = current.replace(month=current.month + 1, day=1)
        period_end = next_d - timedelta(days=1)
        
        if d > end:
            break
            
        periods.append({
            'start': d.strftime('%Y-%m-%d'),
            'end': period_end.strftime('%Y-%m-%d'),
            'name': d.strftime('%Y-%m-%d')
        })
    
    # Next month
    if current.month == 12:
        current = current.replace(year=current.year + 1, month=1)
    else:
        current = current.replace(month=current.month + 1)

print(json.dumps(periods))
