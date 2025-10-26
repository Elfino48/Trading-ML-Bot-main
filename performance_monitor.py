import psutil
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List
import pandas as pd
import numpy as np

class PerformanceMonitor:
    def __init__(self):
        self.cpu_readings = []
        self.memory_readings = []
        self.api_latencies = []
        self.cycle_times = []
        self.start_time = time.time()
        self.monitoring = False
        self.monitor_thread = None
        
    def start_monitoring(self, interval: int = 10):
        """Start background monitoring"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, args=(interval,), daemon=True)
        self.monitor_thread.start()
        print("📊 Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        print("📊 Performance monitoring stopped")
    
    def _monitor_loop(self, interval: int):
        """Background monitoring loop"""
        while self.monitoring:
            try:
                cpu = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory().percent
                
                self.cpu_readings.append(cpu)
                self.memory_readings.append(memory)
                
                # Keep only last hour of data
                one_hour_ago = time.time() - 3600
                self._clean_old_data(one_hour_ago)
                
                time.sleep(interval)
            except Exception as e:
                print(f"❌ Error in monitoring loop: {e}")
                time.sleep(interval)
    
    def _clean_old_data(self, cutoff_time: float):
        """Clean old monitoring data"""
        # For simplicity, we'll just keep last 100 readings
        max_readings = 100
        if len(self.cpu_readings) > max_readings:
            self.cpu_readings = self.cpu_readings[-max_readings:]
            self.memory_readings = self.memory_readings[-max_readings:]
            self.api_latencies = self.api_latencies[-max_readings:]
            self.cycle_times = self.cycle_times[-max_readings:]
    
    def record_api_latency(self, latency: float):
        """Record API call latency"""
        self.api_latencies.append(latency)
    
    def record_cycle_time(self, cycle_time: float):
        """Record trading cycle time"""
        self.cycle_times.append(cycle_time)
    
    def check_resources(self, max_cpu: float = 80, max_memory: float = 85) -> bool:
        """Check if system resources are within limits"""
        try:
            cpu = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory().percent
            
            if cpu > max_cpu:
                print(f"🛑 High CPU usage: {cpu}%")
                return False
            if memory > max_memory:
                print(f"🛑 High memory usage: {memory}%")
                return False
                
            return True
        except Exception as e:
            print(f"❌ Error checking resources: {e}")
            return True  # Default to True to avoid stopping on monitoring errors
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary"""
        if not self.cpu_readings:
            return {}
            
        return {
            'current_cpu': self.cpu_readings[-1] if self.cpu_readings else 0,
            'current_memory': self.memory_readings[-1] if self.memory_readings else 0,
            'avg_cpu': np.mean(self.cpu_readings) if self.cpu_readings else 0,
            'avg_memory': np.mean(self.memory_readings) if self.memory_readings else 0,
            'max_cpu': max(self.cpu_readings) if self.cpu_readings else 0,
            'max_memory': max(self.memory_readings) if self.memory_readings else 0,
            'avg_api_latency': np.mean(self.api_latencies) if self.api_latencies else 0,
            'avg_cycle_time': np.mean(self.cycle_times) if self.cycle_times else 0,
            'uptime_hours': (time.time() - self.start_time) / 3600
        }
    
    def get_health_status(self) -> Dict:
        """Get system health status"""
        summary = self.get_performance_summary()
        
        health_checks = {
            'cpu_ok': summary.get('current_cpu', 0) < 80,
            'memory_ok': summary.get('current_memory', 0) < 85,
            'api_latency_ok': summary.get('avg_api_latency', 0) < 2.0,
            'cycle_time_ok': summary.get('avg_cycle_time', 0) < 300  # 5 minutes
        }
        
        health_status = all(health_checks.values())
        
        return {
            'healthy': health_status,
            'checks': health_checks,
            'summary': summary
        }

class DataCache:
    """Simple data caching system"""
    def __init__(self, cache_dir: str = "./cache", ttl: int = 300):
        import diskcache
        self.cache = diskcache.Cache(cache_dir)
        self.ttl = ttl  # Time to live in seconds
    
    def get(self, key: str):
        """Get cached value"""
        try:
            return self.cache.get(key)
        except:
            return None
    
    def set(self, key: str, value, ttl: int = None):
        """Set cached value"""
        try:
            self.cache.set(key, value, expire=ttl or self.ttl)
            return True
        except Exception as e:
            print(f"❌ Cache set error: {e}")
            return False
    
    def clear(self):
        """Clear cache"""
        try:
            self.cache.clear()
            return True
        except:
            return False