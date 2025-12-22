from flask import Blueprint, jsonify, request
from typing import Dict, Any
from api.simulation_manager import SimulationManager
from api.utils import serialize_car

api_vehicle = Blueprint('vehicle', __name__, url_prefix='/api/vehicle')
manager = SimulationManager()

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
    initial_state = data.get('initial_state')

    # Session management
    session_id = manager.create_session(initial_state)

    return jsonify({
        'session_id': session_id,
        'status': 'created',
        'message': f'Session {session_id} created successfully'
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

    # Session Query
    session = manager.get_session(session_id)

    if session is None:
        return jsonify({
            'error': f'Session {session_id} not found!'
        }), 404

    state = session.get_state()
    if state is None:
        return jsonify({
            'session_id': session_id,
            'status': f'Session {session_id} initializing.'
        }), 200
    
    timestamp, car = state
    # Now returning mock data
    return jsonify({
        'session_id': session_id,
        'status': 'active',
        'car_state': serialize_car(car),
        'timestamp': timestamp,
        'message': 'Session status retrieved'
    }), 200

@api_vehicle.route('/session/<session_id>/map', methods=['GET'])
def get_session_map(session_id: str):
    """
    Get map data for visualization
    
    Args:
        session_id: Session ID
    
    Returns:
        JSON: {
            "bounding_box": [xmin, ymin, xmax, ymax],
            "known_obstacles": [[x, y], ...],
            "unknown_obstacles": [[x, y], ...]
        }
    """
    user_session = manager.get_session(session_id)
    if user_session is None:
        return jsonify({
            'error': f'Session {session_id} not found!'
        }), 404
    
    map_data = user_session.get_map_data()
    return jsonify({
        'session_id': session_id,
        **map_data
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

    success = manager.delete_session(session_id)

    if not success:
        return jsonify({
            'error': f'Session {session_id} not found.'
        }), 404
    else:
        return jsonify({
            'message': f'Session {session_id} deleted successfully'
        }), 200