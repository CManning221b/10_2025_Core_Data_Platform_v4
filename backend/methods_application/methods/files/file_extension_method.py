# backend/methods_application/methods/file_extension_method.py
from backend.methods_application.method_implementation import MethodImplementation
import re
from datetime import datetime


class FileExtensionMethod(MethodImplementation):
    method_id = "FileExtensionMethod"
    method_name = "Extract File Extensions"
    description = "Extracts extensions from filenames and creates extension nodes linked to files"

    def execute(self, graph_manager, parameters):
        changes = {
            "nodes_added": 0,
            "edges_added": 0,
            "method_instances_created": 0,
            "processed_files": 0,
            "extensions_added": 0,
            "extension_counts": {}
        }

        # Find file nodes that need processing
        files_to_process = []
        for node_id, node_data in graph_manager.node_data.items():
            if node_data.get('type') == 'file':
                # Check if file already has an extension
                has_extension = False
                for (source, target), edge_attrs in graph_manager.edge_data.items():
                    if source == node_id and edge_attrs.get('edge_type') == 'has_extension':
                        has_extension = True
                        break

                if not has_extension:
                    files_to_process.append(node_id)

        print(f" Found {len(files_to_process)} files that need extension processing")

        # Early exit if no work to do
        if not files_to_process:
            print(" No files need extension processing - skipping method execution")
            return changes

        # Only create method instance if there's actual work to do
        method_instance_id = self._create_method_instance(graph_manager, changes)

        for file_id in files_to_process:
            changes["processed_files"] += 1

            # Find filename nodes connected to this file
            filename_nodes = []
            for (source, target), edge_attrs in graph_manager.edge_data.items():
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
                    print(f" Processing extension '{extension}' from filename '{filename_str}'")

                    # Find existing extension node or create new one
                    existing_ext_node = None
                    for node_id, node_data in graph_manager.node_data.items():
                        if (node_data.get('type') == 'extension' and
                                node_data.get('value') == f".{extension}"):
                            existing_ext_node = node_id
                            break

                    if existing_ext_node:
                        ext_id = existing_ext_node
                    else:
                        # Create new extension node
                        ext_id = self._get_next_numeric_id(graph_manager)

                        try:
                            graph_manager.add_node(
                                node_id=ext_id,
                                value=f".{extension}",
                                type='extension',
                                hierarchy='metadata',
                                attributes={
                                    'extension_string': extension,
                                    'created': datetime.now().isoformat()
                                }
                            )
                            changes["nodes_added"] += 1
                            print(f" Created new extension node {ext_id}")

                            # Connect method to the extension it created
                            self._connect_method_to_output(graph_manager, method_instance_id, ext_id, changes)

                        except KeyError as e:
                            print(f" Error creating extension node: {e}")
                            continue

                    # Connect method to the file it analyzed
                    self._connect_method_to_input(graph_manager, method_instance_id, file_id, changes)

                    # Create has_extension edge
                    self._connect_file_to_extension(graph_manager, file_id, ext_id, changes)

                    changes["extensions_added"] += 1
                    if extension in changes["extension_counts"]:
                        changes["extension_counts"][extension] += 1
                    else:
                        changes["extension_counts"][extension] = 1

                    break  # Only process first filename per file

        print(
            f" FileExtensionMethod processed {changes['processed_files']} files, added {changes['extensions_added']} extensions")
        return changes

    def _create_method_instance(self, graph_manager, changes):
        """Create a method instance node for execution traceability"""
        method_instance_id = self._get_next_numeric_id(graph_manager)

        try:
            graph_manager.add_node(
                node_id=method_instance_id,
                value=f"FileExtensionMethod execution",
                type='FileExtensionMethod',
                hierarchy='analysis',
                attributes={
                    'method_type': 'FileExtensionMethod',
                    'execution_time': datetime.now().isoformat(),
                    'method_id': self.method_id,
                    'method_name': self.method_name
                }
            )
            changes["nodes_added"] += 1
            changes["method_instances_created"] += 1
            print(f" Created method_instance node {method_instance_id}")

        except KeyError as e:
            print(f" Method instance node {method_instance_id} already exists")

        return method_instance_id

    def _connect_method_to_output(self, graph_manager, method_instance_id, ext_id, changes):
        """Connect method instance to the extension it created"""
        print(f" Connecting method {method_instance_id} -> extension {ext_id}")

        try:
            graph_manager.add_edge(
                source=method_instance_id,
                target=ext_id,
                attributes={
                    'edge_type': 'produces',
                    'direction': 'out',
                    'provenance_type': 'generated_by',
                    'created': datetime.now().isoformat()
                }
            )
            changes["edges_added"] += 1
            print(f" Created output edge {method_instance_id} -> {ext_id}")

        except (KeyError, ValueError) as e:
            print(f" Error creating output edge: {e}")

    def _connect_method_to_input(self, graph_manager, method_instance_id, file_id, changes):
        """Connect method instance to the file it analyzed"""
        print(f" Connecting method {method_instance_id} -> file {file_id}")

        try:
            graph_manager.add_edge(
                source=method_instance_id,
                target=file_id,
                attributes={
                    'edge_type': 'uses',
                    'direction': 'out',
                    'provenance_type': 'derived_from',
                    'created': datetime.now().isoformat()
                }
            )
            changes["edges_added"] += 1
            print(f" Created input edge {method_instance_id} -> {file_id}")

        except (KeyError, ValueError) as e:
            print(f" Error creating input edge: {e}")

    def _connect_file_to_extension(self, graph_manager, file_id, ext_id, changes):
        """Connect file to its extension"""
        print(f" Connecting file {file_id} -> extension {ext_id}")

        try:
            graph_manager.add_edge(
                source=file_id,
                target=ext_id,
                attributes={
                    'edge_type': 'has_extension',
                    'direction': 'out',
                    'provenance_type': 'contains_metadata',
                    'created': datetime.now().isoformat()
                }
            )
            changes["edges_added"] += 1
            print(f" Created file-extension edge {file_id} -> {ext_id}")

        except (KeyError, ValueError) as e:
            print(f" Error creating file-extension edge: {e}")

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

        result = str(next_id)
        print(f" Generated next ID: {result}")
        return result

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