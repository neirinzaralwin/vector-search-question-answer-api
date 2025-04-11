import os
from flask import Flask
from controllers import search_controller, index_controller, health_controller, qa_controller

def create_app():
    app = Flask(__name__)
    
    # Initialize services
    from services.database_service import init_database
    from services.index_service import init_index_service
    from services.embedding_service import init_embedding_model
    from services.qa_service import init_qa_service
    
    init_database()
    init_embedding_model()
    init_index_service()
    init_qa_service()
    
    # Register blueprints
    app.register_blueprint(search_controller.bp)
    app.register_blueprint(index_controller.bp)
    app.register_blueprint(health_controller.bp)
    app.register_blueprint(qa_controller.bp)
    
    return app

if __name__ == '__main__':
    app = create_app()
    port = int(os.environ.get('PORT', 3032))
    app.run(host='0.0.0.0', port=port, debug=False)