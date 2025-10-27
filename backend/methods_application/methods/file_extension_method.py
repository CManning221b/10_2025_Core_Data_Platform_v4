# backend/methods_application/methods/file_extension_method.py
from backend.methods_application.method_implementation import MethodImplementation
import re


class FileExtensionMethod(MethodImplementation):
    method_id = "FileExtensionMethod"
    method_name = "Extract File Extensions"
    description = "Extracts extensions from filenames and creates extension nodes linked to files"

    def execute(self, graph_manager, parameters):
        changes = {
            "nodes_added": 0,
            "edges_added": 0,
            "processed_files": 0,
            "extensions_added": 0,
            "extension_counts": {}
        }

        # Find file nodes by directly searching node_data
        file_nodes = []
        for node_id, node_data in graph_manager.node_data.items():
            if node_data.get('type') == 'file':
                file_nodes.append(node_id)

        for file_id in file_nodes:
            changes["processed_files"] += 1

            # Check if file already has an extension by looking at edges
            has_extension = False
            for edge_id, edge_attrs in graph_manager.edge_data.items():
                if '_' in edge_id:
                    source, target = edge_id.split('_', 1)
                    if source == file_id and edge_attrs.get('edge_type') == 'has_extension':
                        has_extension = True
                        break

            if has_extension:
                continue  # Skip if extension already exists

            # Find filename nodes connected to this file
            filename_nodes = []
            for edge_id, edge_attrs in graph_manager.edge_data.items():
                if '_' in edge_id:
                    source, target = edge_id.split('_', 1)
                    if (source == file_id and edge_attrs.get('edge_type') == 'has_name'):
                        filename_nodes.append(target)

            for filename_id in filename_nodes:
                filename_node = graph_manager.node_data.get(filename_id)
                if not filename_node:
                    continue

                # Extract extension from filename
                filename_str = filename_node.get('filename_string', '') or filename_node.get('value', '')
                extension = self._extract_extension(filename_str)

                if extension and self._is_recognized_extension(extension):
                    # Find existing extension node or create new one
                    existing_ext_node = None

                    # Search for existing extension node
                    for node_id, node_data in graph_manager.node_data.items():
                        if (node_data.get('type') == 'extension' and
                                node_data.get('value') == f".{extension}"):
                            existing_ext_node = node_id
                            break

                    if existing_ext_node:
                        ext_id = existing_ext_node
                    else:
                        # Create new extension node with consistent numeric ID
                        ext_id = self._get_next_numeric_id(graph_manager)

                        # Add node directly to node_data
                        graph_manager.node_data[ext_id] = {
                            'value': f".{extension}",
                            'type': 'extension',
                            'hierarchy_level': None
                        }
                        changes["nodes_added"] += 1

                    # Create has_extension edge using consistent format
                    edge_id = f"{file_id}_{ext_id}"
                    if edge_id not in graph_manager.edge_data:
                        graph_manager.edge_data[edge_id] = {
                            'edge_type': 'has_extension',
                            'direction': 'out',
                            'source_type': 'file',
                            'target_type': 'extension'
                        }
                        changes["edges_added"] += 1
                        changes["extensions_added"] += 1

                        # Count extensions
                        if extension in changes["extension_counts"]:
                            changes["extension_counts"][extension] += 1
                        else:
                            changes["extension_counts"][extension] = 1

                    break  # Only process first filename per file

        # Mark graph as inconsistent so caches get rebuilt
        graph_manager._mark_inconsistent()

        return changes

    def _get_next_numeric_id(self, graph_manager):
        """Generate next available numeric ID"""
        existing_ids = set()
        for node_id in graph_manager.node_data.keys():
            try:
                existing_ids.add(int(node_id))
            except (ValueError, TypeError):
                pass

        next_id = 0
        while next_id in existing_ids:
            next_id += 1
        return str(next_id)

    def _extract_extension(self, filename):
        """Extract extension from filename string"""
        if not filename:
            return None

        match = re.search(r'\.([a-zA-Z0-9]+)$', filename.strip())
        return match.group(1).lower() if match else None

    def _is_recognized_extension(self, extension):
        """Check if extension is in our recognized list"""
        recognized_extensions = {
            'pdf', 'doc', 'docx', 'txt', 'rtf', 'odt',
            'jpg', 'jpeg', 'png', 'gif', 'bmp', 'svg', 'tiff',
            'mp3', 'wav', 'flac', 'aac', 'ogg',
            'mp4', 'avi', 'mkv', 'mov', 'wmv', 'flv',
            'py', 'js', 'html', 'css', 'java', 'cpp', 'c', 'php',
            'json', 'xml', 'csv', 'xlsx', 'sql',
            'zip', 'rar', '7z', 'tar', 'gz', 'pptx'
        }
        return extension.lower() in recognized_extensions