from flask import Blueprint, jsonify, request
from typing import Dict, Any

api_vehicle = Blueprint('vehicle', __name__, url_prefix='/api/vehicle')

@api_vehicle.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint
    Used to verify if the API service is running properly

    Returns:
        JSON: {"status": "ok", "message": "API is running"}
    """
    return jsonify({
        'status': 'ok',
        'message': 'API is running'
    }), 200

@api_vehicle.route('/session/create', methods=['POST'])
def create_session():
    """
    Create a new simulation session

    Request body (optional):
        {
            "initial_state": {
                "x": 5.0,
                "y": 5.0,
                "yaw": 0.0
            }
        }

    Returns:
        JSON: {
            "session_id": "abc123",
            "status": "created"
        }
    """

    # acquire JSON sent by clients
    data = request.get_json() or {}

    # TODO： Session management
    # Use mock session_id for now
    import uuid
    session_id = str(uuid.uuid4())

    return jsonify({
        'session_id': session_id,
        'status': 'created',
        'message': 'Session created successfully'
    }), 201


@api_vehicle.route('/session/<session_id>/status', methods=['GET'])
def get_session_status(session_id: str):
    """
    Query session status

    Args:
        session_id: Session ID (obtained from the URL path)
    
    Returns:
        JSON: {
            "session_id": "abc123",
            "status": "active",
            "car_state": {...}
        }
    """

    # TODO: Session Query
    # Now returning mock data
    return jsonify({
        'session_id': session_id,
        'status': 'active',
        'message': 'Session status retrieved'
    }), 200

@api_vehicle.route('/session/<session_id>', methods=['DELETE'])
def delete_session(session_id: str):
    """
    Delete a session

    Args:
        session_id: Session ID

    Returns:
        JSON: {"message": "Session deleted"}
    """

    # TODO: Implement real session deletion later
    return jsonify({
        'message': f'Session {session_id} deleted successfully'
    }), 200