# Healvana Integration API Documentation

Version: 1.0.0  
Last Updated: May 14, 2025

## Overview

The Healvana Integration API provides a secure bridge between client applications and the Healvana Chat system for mental health support. This API handles authentication, session management, user context, clinical safety monitoring, and compliance requirements while abstracting the underlying Healvana Chat API.

## Base URL

```
https://api.yourdomain.com/mental-health/v1
```

## Authentication

All API requests require authentication using JWT Bearer tokens.

```
Authorization: Bearer <your_jwt_token>
```

### Obtaining an Access Token

```http
POST /auth/token
Content-Type: application/json

{
  "client_id": "your_client_id",
  "client_secret": "your_client_secret"
}
```

#### Response

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "Bearer",
  "expires_in": 3600,
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

### Refreshing a Token

```http
POST /auth/refresh
Content-Type: application/json

{
  "refresh_token": "your_refresh_token"
}
```

## User Management

### Create User Profile

```http
POST /users
Content-Type: application/json
Authorization: Bearer <token>

{
  "external_user_id": "user-123",
  "preferred_locale": "en-US",
  "clinical_context": {
    "risk_level": "low",
    "care_plan_type": "general",
    "therapy_focus": ["anxiety", "stress-management"],
    "emergency_contact": {
      "name": "John Doe",
      "relationship": "Partner",
      "phone": "+15551234567"
    }
  },
  "preferences": {
    "message_style": "supportive",
    "interaction_frequency": "daily"
  }
}
```

#### Response

```json
{
  "user_id": "hv-84729ed8-8374",
  "external_user_id": "user-123",
  "session_id": "session-94829382a",
  "created_at": "2025-05-14T12:00:00Z",
  "status": "active"
}
```

### Get User Profile

```http
GET /users/{user_id}
Authorization: Bearer <token>
```

#### Response

```json
{
  "user_id": "hv-84729ed8-8374",
  "external_user_id": "user-123",
  "preferred_locale": "en-US",
  "clinical_context": {
    "risk_level": "low",
    "care_plan_type": "general",
    "therapy_focus": ["anxiety", "stress-management"]
  },
  "preferences": {
    "message_style": "supportive",
    "interaction_frequency": "daily"
  },
  "session_id": "session-94829382a",
  "created_at": "2025-05-14T12:00:00Z",
  "last_interaction": "2025-05-14T14:30:00Z",
  "status": "active"
}
```

### Update User Profile

```http
PATCH /users/{user_id}
Content-Type: application/json
Authorization: Bearer <token>

{
  "preferred_locale": "en-GB",
  "clinical_context": {
    "therapy_focus": ["anxiety", "stress-management", "sleep"]
  }
}
```

## Conversation Management

### Start Conversation

```http
POST /conversations
Content-Type: application/json
Authorization: Bearer <token>

{
  "user_id": "hv-84729ed8-8374"
}
```

#### Response

```json
{
  "conversation_id": "conv-3472938",
  "user_id": "hv-84729ed8-8374",
  "started_at": "2025-05-14T15:00:00Z",
  "greeting": "Hello! I'm Healvana, your mental wellness companion for the US. I'd like to understand how you've been feeling. How have you been lately?"
}
```

### Send Message

```http
POST /conversations/{conversation_id}/messages
Content-Type: application/json
Authorization: Bearer <token>

{
  "message": "I've been feeling anxious about work lately.",
  "metadata": {
    "client_message_id": "client-msg-123",
    "client_timestamp": "2025-05-14T15:01:12Z"
  }
}
```

#### Response

```json
{
  "message_id": "msg-493872",
  "conversation_id": "conv-3472938",
  "user_id": "hv-84729ed8-8374",
  "type": "user",
  "content": "I've been feeling anxious about work lately.",
  "timestamp": "2025-05-14T15:01:15Z",
  "metadata": {
    "client_message_id": "client-msg-123",
    "client_timestamp": "2025-05-14T15:01:12Z"
  }
}
```

### Stream Response

```http
GET /conversations/{conversation_id}/messages/stream
Authorization: Bearer <token>
Accept: text/event-stream
```

The response is streamed as a series of Server-Sent Events:

```
event: message_start
data: {"message_id":"msg-493873","conversation_id":"conv-3472938","type":"ai","timestamp":"2025-05-14T15:01:16Z"}

event: token
data: {"message_id":"msg-493873","token":"Can"}

event: token
data: {"message_id":"msg-493873","token":" you"}

event: token
data: {"message_id":"msg-493873","token":" tell"}

event: token
data: {"message_id":"msg-493873","token":" me"}

event: token
data: {"message_id":"msg-493873","token":" more"}

event: token
data: {"message_id":"msg-493873","token":" about"}

event: token
data: {"message_id":"msg-493873","token":" what's"}

event: token
data: {"message_id":"msg-493873","token":" happening"}

event: token
data: {"message_id":"msg-493873","token":" at"}

event: token
data: {"message_id":"msg-493873","token":" work?"}

event: message_complete
data: {"message_id":"msg-493873","conversation_id":"conv-3472938","type":"ai","content":"Can you tell me more about what's happening at work?","timestamp":"2025-05-14T15:01:18Z"}
```

### Get Conversation History

```http
GET /conversations/{conversation_id}/messages
Authorization: Bearer <token>
```

#### Response

```json
{
  "conversation_id": "conv-3472938",
  "user_id": "hv-84729ed8-8374",
  "messages": [
    {
      "message_id": "msg-493871",
      "type": "ai",
      "content": "Hello! I'm Healvana, your mental wellness companion for the US. I'd like to understand how you've been feeling. How have you been lately?",
      "timestamp": "2025-05-14T15:00:00Z"
    },
    {
      "message_id": "msg-493872",
      "type": "user",
      "content": "I've been feeling anxious about work lately.",
      "timestamp": "2025-05-14T15:01:15Z"
    },
    {
      "message_id": "msg-493873",
      "type": "ai",
      "content": "Can you tell me more about what's happening at work?",
      "timestamp": "2025-05-14T15:01:18Z"
    }
  ],
  "started_at": "2025-05-14T15:00:00Z",
  "last_message_at": "2025-05-14T15:01:18Z"
}
```

### End Conversation

```http
POST /conversations/{conversation_id}/end
Authorization: Bearer <token>
```

#### Response

```json
{
  "conversation_id": "conv-3472938",
  "user_id": "hv-84729ed8-8374",
  "started_at": "2025-05-14T15:00:00Z",
  "ended_at": "2025-05-14T15:30:00Z",
  "status": "completed",
  "message_count": 10
}
```

## Clinical Insights

### Get Mood Analysis

```http
GET /users/{user_id}/insights/mood
Authorization: Bearer <token>
```

#### Response

```json
{
  "user_id": "hv-84729ed8-8374",
  "analysis_period": {
    "start": "2025-04-14T00:00:00Z",
    "end": "2025-05-14T23:59:59Z"
  },
  "mood_trends": [
    {
      "date": "2025-05-14",
      "primary_mood": "anxious",
      "secondary_mood": "stressed",
      "intensity": 0.7,
      "context": ["work", "deadlines"]
    },
    {
      "date": "2025-05-13",
      "primary_mood": "anxious",
      "secondary_mood": "tired",
      "intensity": 0.6,
      "context": ["work", "sleep"]
    }
  ],
  "common_triggers": [
    {
      "trigger": "work deadlines",
      "frequency": 0.8,
      "associated_moods": ["anxious", "stressed"]
    },
    {
      "trigger": "interpersonal conflict",
      "frequency": 0.3,
      "associated_moods": ["sad", "frustrated"]
    }
  ],
  "recommended_coping_strategies": [
    {
      "strategy": "deep breathing",
      "relevance_score": 0.9,
      "context": "For immediate anxiety relief"
    },
    {
      "strategy": "task prioritization",
      "relevance_score": 0.8,
      "context": "For work-related stress"
    }
  ]
}
```

### Get Risk Assessment

Restricted to authorized clinical staff only.

```http
GET /users/{user_id}/insights/risk
Authorization: Bearer <token>
X-Clinical-Role: therapist
```

#### Response

```json
{
  "user_id": "hv-84729ed8-8374",
  "current_risk_level": "low",
  "last_assessment": "2025-05-14T16:00:00Z",
  "risk_factors": [
    {
      "factor": "sleep disturbance",
      "severity": "moderate",
      "mentioned_at": "2025-05-13T14:22:30Z",
      "context": "Reported difficulty falling asleep due to work thoughts"
    }
  ],
  "protective_factors": [
    {
      "factor": "social support",
      "strength": "strong",
      "mentioned_at": "2025-05-12T11:15:22Z",
      "context": "Mentioned supportive partner and weekly calls with friends"
    }
  ],
  "crisis_indicators": {
    "present": false,
    "last_checked": "2025-05-14T16:00:00Z"
  },
  "recommended_actions": [
    {
      "action": "sleep hygiene assessment",
      "priority": "medium"
    }
  ]
}
```

## Administration

### Get System Health

```http
GET /admin/health
Authorization: Bearer <token>
```

#### Response

```json
{
  "status": "healthy",
  "components": {
    "auth_service": "operational",
    "healvana_api": "operational",
    "database": "operational",
    "message_queue": "operational"
  },
  "metrics": {
    "average_response_time": 320,
    "current_active_sessions": 128,
    "error_rate_last_hour": 0.002
  },
  "timestamp": "2025-05-14T17:00:00Z"
}
```

### Get Usage Statistics

```http
GET /admin/usage
Authorization: Bearer <token>
```

#### Response

```json
{
  "period": {
    "start": "2025-05-14T00:00:00Z",
    "end": "2025-05-14T17:00:00Z"
  },
  "active_users": 342,
  "new_users": 28,
  "total_messages": 5621,
  "average_conversation_length": 12,
  "peak_time": "2025-05-14T12:00:00Z",
  "usage_by_locale": [
    {
      "locale": "en-US",
      "users": 289,
      "messages": 4892
    },
    {
      "locale": "en-GB",
      "users": 53,
      "messages": 729
    }
  ]
}
```

## Webhooks

### Register Webhook

```http
POST /webhooks
Content-Type: application/json
Authorization: Bearer <token>

{
  "url": "https://yourapplication.com/webhook/healvana",
  "events": ["message.created", "risk.elevated", "session.ended"],
  "secret": "your-webhook-secret"
}
```

#### Response

```json
{
  "webhook_id": "wh-9482394",
  "url": "https://yourapplication.com/webhook/healvana",
  "events": ["message.created", "risk.elevated", "session.ended"],
  "created_at": "2025-05-14T17:30:00Z",
  "status": "active"
}
```

### Webhook Payload Examples

#### `message.created` Event

```json
{
  "event": "message.created",
  "timestamp": "2025-05-14T17:35:00Z",
  "data": {
    "message_id": "msg-493900",
    "conversation_id": "conv-3472938",
    "user_id": "hv-84729ed8-8374",
    "type": "ai",
    "content": "What coping strategies have you tried so far?",
    "created_at": "2025-05-14T17:35:00Z"
  }
}
```

#### `risk.elevated` Event

```json
{
  "event": "risk.elevated",
  "timestamp": "2025-05-14T18:00:00Z",
  "data": {
    "user_id": "hv-84729ed8-8374",
    "conversation_id": "conv-3472938",
    "previous_risk_level": "low",
    "current_risk_level": "moderate",
    "risk_factors": [
      {
        "factor": "sleep disturbance",
        "severity": "significant",
        "context": "Reported not sleeping for 48 hours"
      }
    ],
    "triggering_message_id": "msg-493920",
    "recommended_actions": [
      {
        "action": "sleep assessment",
        "priority": "high"
      }
    ]
  }
}
```

## Error Codes

| Code | Description |
|------|-------------|
| 400  | Bad Request - The request was malformed or missing required parameters |
| 401  | Unauthorized - Authentication is required or the provided credentials are invalid |
| 403  | Forbidden - The authenticated user doesn't have permission to access the resource |
| 404  | Not Found - The requested resource doesn't exist |
| 409  | Conflict - The request conflicts with the current state of the resource |
| 422  | Unprocessable Entity - The request was well-formed but contains semantic errors |
| 429  | Too Many Requests - Rate limit exceeded |
| 500  | Internal Server Error - Something went wrong on the server |
| 503  | Service Unavailable - The service is temporarily unavailable |

### Error Response Format

```json
{
  "error": {
    "code": "invalid_request",
    "message": "The request was invalid",
    "details": [
      {
        "field": "preferred_locale",
        "issue": "must be one of the supported locales: en-US, en-GB"
      }
    ],
    "request_id": "req-3928392"
  }
}
```

## Rate Limiting

The API implements rate limiting to ensure system stability. Rate limits vary by endpoint and client tier:

- Standard tier: 60 requests per minute
- Professional tier: 300 requests per minute
- Enterprise tier: Custom limits

Rate limit headers are included in all responses:

```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 58
X-RateLimit-Reset: 1589305200
```

When a rate limit is exceeded, the API responds with a 429 status code.

## Security Considerations

### Data Privacy

- All data is encrypted in transit using TLS 1.3
- All sensitive data is encrypted at rest using AES-256
- PII is tokenized where possible
- Mental health data is handled in compliance with relevant regulations (HIPAA, GDPR, etc.)

### Clinical Safety

- Automatic risk monitoring is implemented on all conversations
- High-risk indicators trigger immediate alerts to clinical staff
- Crisis protocols are in place for emergency situations

## SDK Examples

### JavaScript/TypeScript

```typescript
import { HealvanaClient } from '@healvana/client';

// Initialize client
const client = new HealvanaClient({
  apiKey: 'your_api_key',
  baseUrl: 'https://api.yourdomain.com/mental-health/v1'
});

// Create a user
const user = await client.users.create({
  external_user_id: 'user-123',
  preferred_locale: 'en-US',
  clinical_context: {
    risk_level: 'low',
    therapy_focus: ['anxiety', 'stress-management']
  }
});

// Start a conversation
const conversation = await client.conversations.start({
  user_id: user.user_id
});

// Send a message
const message = await client.messages.send({
  conversation_id: conversation.conversation_id,
  message: "I've been feeling anxious about work lately."
});

// Stream a response
const stream = await client.messages.stream({
  conversation_id: conversation.conversation_id
});

stream.on('token', (token) => {
  console.log(token);
});

stream.on('complete', (message) => {
  console.log('Complete message:', message.content);
});
```

### Python

```python
from healvana import HealvanaClient

# Initialize client
client = HealvanaClient(
    api_key='your_api_key',
    base_url='https://api.yourdomain.com/mental-health/v1'
)

# Create a user
user = client.users.create(
    external_user_id='user-123',
    preferred_locale='en-US',
    clinical_context={
        'risk_level': 'low',
        'therapy_focus': ['anxiety', 'stress-management']
    }
)

# Start a conversation
conversation = client.conversations.start(
    user_id=user['user_id']
)

# Send a message
message = client.messages.send(
    conversation_id=conversation['conversation_id'],
    message="I've been feeling anxious about work lately."
)

# Stream a response
for event in client.messages.stream(
    conversation_id=conversation['conversation_id']
):
    if event['type'] == 'token':
        print(event['token'], end='')
    elif event['type'] == 'complete':
        print('\nComplete message:', event['message']['content'])
```

## Changelog

### v1.0.0 (2025-05-14)

- Initial release of the Healvana Integration API
- Core conversation functionality
- User management
- Clinical insights
- Webhook support
