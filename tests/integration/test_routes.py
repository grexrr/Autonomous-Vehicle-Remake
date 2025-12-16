from api.app import create_app

if __name__ == '__main__':
    print("[Testing][Routes Registration]...\n")
    app = create_app('development')
    
    # Print all registered routes
    print("Registered routes:")
    for rule in app.url_map.iter_rules():
        print(f"  {rule.methods} {rule.rule}")
    
    print("\n✅ Routes registered successfully!")
    print("\nYou can start the server to test:")
    print("  from api.app import app, socketio")
    print("  socketio.run(app, host='0.0.0.0', port=5001, debug=True)")
