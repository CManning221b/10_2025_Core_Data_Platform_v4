# app/app.py - modify this file to register the main blueprint
from flask import Flask, render_template


def create_app(config_name='development'):
    # Initialize Flask app
    app = Flask(__name__)

    # Load configuration
    from config import config_by_name
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
    app.config['JSON_AS_ASCII'] = False
    app.config.from_object(config_by_name[config_name])

    # Register blueprints
    from routes.main_routes import main_bp
    from routes.upload_routes import upload_bp
    from routes.instance_routes import instance_bp
    from routes.ontology_routes import ontology_bp
    from routes.process_routes import process_bp
    from routes.reasoning_routes import reasoning_bp
    from routes.visualisation_routes import visualisation_bp
    from routes.text_interface_routes import text_interface_bp
    from routes.graph_propagation_routes import propagation_bp

    from api.visualisation_api import visualisation_api
    from api.ontology_api import ontology_api


    # Register any other blueprints

    app.register_blueprint(main_bp)
    app.register_blueprint(upload_bp)
    app.register_blueprint(instance_bp)
    app.register_blueprint(ontology_bp)
    app.register_blueprint(process_bp)
    app.register_blueprint(reasoning_bp)
    app.register_blueprint(text_interface_bp)
    app.register_blueprint(visualisation_bp)
    app.register_blueprint(propagation_bp)
    app.register_blueprint(visualisation_api)
    app.register_blueprint(ontology_api)


    # Register other blueprints

    # Setup error handlers
    @app.errorhandler(404)
    def page_not_found(e):
        return render_template('errors/404.html'), 404

    @app.errorhandler(500)
    def internal_server_error(e):
        return render_template('errors/500.html'), 500

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(host='127.0.0.3', port=5002, debug=app.config['DEBUG'])
    # app.run(host='0.0.0.0', port=5000, debug=app.config['DEBUG'])


