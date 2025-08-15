# app/core/redis_client.py
import os
import redis

REDIS_URL = os.getenv("REDIS_URL", "redis://redis-19413.c15.us-east-1-4.ec2.redns.redis-cloud.com:19413")

redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
