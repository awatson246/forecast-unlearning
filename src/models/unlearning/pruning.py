from sklearn.metrics import root_mean_squared_error
import lightgbm as lgb
import json
import re
from collections import defaultdict
from collections import Counter

def prune_lightgbm_trees(model, masked_indices, feature_names, testX, testY, pruning_type):
    """
    Prune trees in a LightGBM model based on pruning_type and evaluate the model.
    """
    def adjust_indices(masked_indices, look_back, n_features):
        """
        Adjust the masked indices to account for the flattened feature space.
        Each original feature will be repeated 'look_back' times in the flattened array.
        """
        adjusted_indices = []
        for idx in masked_indices:
            adjusted_indices.extend([idx + i * n_features for i in range(look_back)])
        return adjusted_indices
    
    def count_leaves(node):
        """
        Recursively count the number of leaf nodes in a tree.
        """
        if "split_index" not in node:  # Leaf node check
            return 1
        left_count = count_leaves(node.get("left_child", {}))
        right_count = count_leaves(node.get("right_child", {}))
        return left_count + right_count
    
    def recover_metadata(tree, idx):
        """Ensure essential metadata is preserved."""
        defaults = {
            "tree_index": idx,
            "num_leaves": 0,
            "num_cat": 0,
            "shrinkage": 0.1,
        }
        for key, default in defaults.items():
            tree[key] = tree.get(key, default)
        return tree
    def ensure_metadata_order(tree, idx):
        """Ensure required metadata appears in the correct order."""
        # Define required metadata with default values
        required_metadata = {
            "tree_index": idx,
            "num_leaves": 0,
            "num_cat": 0,
            "shrinkage": 0.1,
        }

        # Extract tree structure if it exists
        tree_structure = tree.pop("tree_structure", {})
        
        # Rebuild the tree with enforced metadata order
        ordered_tree = {
            "tree_index": tree.get("tree_index", required_metadata["tree_index"]),
            "num_leaves": tree.get("num_leaves", required_metadata["num_leaves"]),
            "num_cat": tree.get("num_cat", required_metadata["num_cat"]),
            "shrinkage": tree.get("shrinkage", required_metadata["shrinkage"]),
            "tree_structure": tree_structure,
        }

        return ordered_tree


    def prune_tree_basic(split, adjusted_masked_indices):
        """
        Prune a tree by converting masked split nodes into leaves, retaining structure.
        """
        # Check if it's a leaf node
        if "leaf_index" in split:
            #print(f"Leaf node encountered with leaf_value: {split['leaf_value']}")
            return {
                "leaf_index": split.get("leaf_index", 0),
                "leaf_value": split["leaf_value"],
                "leaf_weight": split.get("leaf_weight", 0),
                "leaf_count": split.get("leaf_count", 0),
            }

        # Check for internal node
        if "split_feature" in split:
            #print(f"Visiting node with split feature {split['split_feature']} and index {split.get('split_index', 'unknown')}")

            if split["split_feature"] in adjusted_masked_indices:
                print(f"Pruning feature {split['split_feature']} at node {split.get('split_index', 'unknown')}")

                left_child = split.get("left_child", {})
                right_child = split.get("right_child", {})

                if "leaf_value" in left_child and "leaf_value" in right_child:
                    pruned_value = (left_child["leaf_value"] + right_child["leaf_value"]) / 2
                    return {
                        "leaf_index": -1,  # New leaf node
                        "leaf_value": pruned_value,
                        "leaf_weight": left_child.get("leaf_weight", 0) + right_child.get("leaf_weight", 0),
                        "leaf_count": left_child.get("leaf_count", 0) + right_child.get("leaf_count", 0),
                    }
                else:
                    print(f"Warning: Unexpected structure at node {split.get('split_index', 'unknown')}")
                    return {
                        "leaf_index": -1,
                        "leaf_value": 0,
                        "leaf_weight": 0,
                        "leaf_count": 0,
                    }

        # Recurse into children if not pruning
        if "left_child" in split:
            split["left_child"] = prune_tree_basic(split["left_child"], adjusted_masked_indices)
        if "right_child" in split:
            split["right_child"] = prune_tree_basic(split["right_child"], adjusted_masked_indices)

        return split


    def prune_tree_average(split, adjusted_masked_indices):
        """
        Prune a tree by averaging child leaf values at masked nodes.
        """
        if "leaf_index" in split:
            return {
                "leaf_index": split.get("leaf_index", 0),
                "leaf_value": split["leaf_value"],
                "leaf_weight": split.get("leaf_weight", 0),
                "leaf_count": split.get("leaf_count", 0),
            }

        if "split_feature" in split:
            if split["split_feature"] in adjusted_masked_indices:
                left_child = split.get("left_child", {})
                right_child = split.get("right_child", {})

                if "leaf_value" in left_child and "leaf_value" in right_child:
                    averaged_value = (left_child["leaf_value"] + right_child["leaf_value"]) / 2
                    return {
                        "leaf_index": -1,
                        "leaf_value": averaged_value,
                        "leaf_weight": left_child.get("leaf_weight", 0) + right_child.get("leaf_weight", 0),
                        "leaf_count": left_child.get("leaf_count", 0) + right_child.get("leaf_count", 0),
                    }

        if "left_child" in split:
            split["left_child"] = prune_tree_average(split["left_child"], adjusted_masked_indices)
        if "right_child" in split:
            split["right_child"] = prune_tree_average(split["right_child"], adjusted_masked_indices)

        return split


    def prune_tree_weighted(split, adjusted_masked_indices):
        """
        Prune a tree by calculating a weighted average of child leaf values.
        """
        if "leaf_index" in split:
            return {
                "leaf_index": split.get("leaf_index", 0),
                "leaf_value": split["leaf_value"],
                "leaf_weight": split.get("leaf_weight", 0),
                "leaf_count": split.get("leaf_count", 0),
            }

        if "split_feature" in split:
            if split["split_feature"] in adjusted_masked_indices:
                left_child = split.get("left_child", {})
                right_child = split.get("right_child", {})

                if "leaf_value" in left_child and "leaf_value" in right_child:
                    left_weight = left_child.get("leaf_count", 1)
                    right_weight = right_child.get("leaf_count", 1)
                    weighted_value = (
                        left_child["leaf_value"] * left_weight + right_child["leaf_value"] * right_weight
                    ) / (left_weight + right_weight)
                    return {
                        "leaf_index": -1,
                        "leaf_value": weighted_value,
                        "leaf_weight": left_child.get("leaf_weight", 0) + right_child.get("leaf_weight", 0),
                        "leaf_count": left_child.get("leaf_count", 0) + right_child.get("leaf_count", 0),
                    }

        if "left_child" in split:
            split["left_child"] = prune_tree_weighted(split["left_child"], adjusted_masked_indices)
        if "right_child" in split:
            split["right_child"] = prune_tree_weighted(split["right_child"], adjusted_masked_indices)

        return split

    def prune_and_process_trees(model, pruning_type, adjusted_masked_indices):
        """
        Prunes trees from the LightGBM model, ensures metadata consistency and order,
        and writes the output to a JSON file including both the tree metadata and non-tree metadata.
        Returns the pruned model in JSON format.
        """
        prune_tree = pruning_methods[pruning_type]

        # Extract original model metadata
        original_model_dump = model.booster_.dump_model()
        model_metadata = {
            "name": original_model_dump.get("name", "tree"),
            "version": original_model_dump.get("version", "v4"),
            "num_class": original_model_dump.get("num_class", 1),
            "num_tree_per_iteration": original_model_dump.get("num_tree_per_iteration", 1),
            "label_index": original_model_dump.get("label_index", 0),
            "max_feature_idx": original_model_dump.get("max_feature_idx", 89),
            "objective": original_model_dump.get("objective", "regression"),
            "average_output": original_model_dump.get("average_output", False),
            "feature_names": original_model_dump.get("feature_names", []),
            "monotone_constraints": original_model_dump.get("monotone_constraints", []),
        }

        # Extract feature_infos
        feature_infos = original_model_dump.get("feature_infos", {})

        # Extract tree info
        tree_info = original_model_dump["tree_info"]
        
        # Process each tree
        retained_trees = []
        for idx, tree in enumerate(tree_info):
            try:
                # Prune the tree structure
                pruned_structure = prune_tree(tree["tree_structure"], adjusted_masked_indices)
                tree["tree_structure"] = pruned_structure
                tree["num_leaves"] = count_leaves(pruned_structure)
            except Exception as e:
                print(f"Error pruning Tree {idx}: {e}")
                # Retain original structure if pruning fails
                tree["tree_structure"] = tree.get("tree_structure", {})
                tree["num_leaves"] = tree.get("num_leaves", 0)

            # Ensure metadata consistency and proper order
            ordered_tree = ensure_metadata_order(tree, idx)
            retained_trees.append(ordered_tree)

        # Validate all trees are properly structured
        for idx, tree in enumerate(retained_trees):
            required_keys = ["tree_index", "num_leaves", "num_cat", "shrinkage", "tree_structure"]
            missing_keys = [key for key in required_keys if key not in tree]
            if missing_keys:
                print(f"Tree {idx} is missing keys: {missing_keys}")

        # Construct final JSON structure including non-tree metadata
        final_json_structure = {
            **model_metadata,  # Include all non-tree metadata
            "feature_infos": feature_infos,  # Add feature_infos here
            "tree_info": retained_trees  # Add pruned trees as "tree_info"
        }

        # Convert final JSON structure to a string (model_str)
        model_str = json.dumps(final_json_structure, indent=4)

        # Write to JSON file (if you want to persist it)
        output_file = "pruned_trees.json"
        try:
            with open(output_file, "w") as file:
                json.dump(final_json_structure, file, indent=4)
            print(f"Pruned trees successfully written to {output_file}")
        except Exception as e:
            print(f"Error writing JSON: {str(e)}")

        # Return the model_str
        return model_str, retained_trees

    def calculate_feature_importance_from_pruned_trees(trees_json):
        """
        Calculates feature importance from a pruned tree structure.
        Assumes `trees_json` is the JSON-like object containing pruned trees.
        """
        feature_importance = defaultdict(int)
        
        # Recursively extract features used in splits
        def extract_splits(tree_structure):
            if isinstance(tree_structure, dict):
                # If this node is a decision node
                if "split_feature" in tree_structure:
                    feature_importance[tree_structure["split_feature"]] += 1
                # Continue searching in left and right children
                if "left_child" in tree_structure:
                    extract_splits(tree_structure["left_child"])
                if "right_child" in tree_structure:
                    extract_splits(tree_structure["right_child"])

        # Iterate through all the trees and extract splits
        for tree in trees_json:
            tree_structure = tree.get("tree_structure", {})
            extract_splits(tree_structure)
        
        # Normalize importance if needed, e.g., divide by the number of trees
        total_trees = len(trees_json)
        for feature in feature_importance:
            feature_importance[feature] /= total_trees  # Normalize by the number of trees

        return feature_importance
    

    def calculate_and_print_feature_importance(model, testX, feature_names, look_back=1):
        """
        Calculates and prints the feature importance for the original model.
        The feature importance is normalized and mapped back to the original feature names.

        Args:
        - model: The original trained model.
        - testX: Test data (reshaped if necessary).
        - feature_names: List of feature names corresponding to the input data.
        - look_back: Number of timesteps to normalize importance (optional).
        """
        # Ensure that test data is flattened if necessary
        if len(testX.shape) > 2:
            testX_reshaped = testX.reshape(testX.shape[0], -1)
        else:
            testX_reshaped = testX

        # Get the feature importance from the model (assuming the model has .feature_importance_ attribute)
        feature_importance = model.feature_importance(importance_type='split')  # You can also use 'gain'

        # Initialize a dictionary to group importance by feature name
        grouped_importances = {name: 0 for name in feature_names}

        # Map feature importance to the original feature names
        for i, importance in enumerate(feature_importance):
            original_feature_idx = i % len(feature_names)  # Handle wrapping around if necessary
            original_feature_name = feature_names[original_feature_idx]
            grouped_importances[original_feature_name] += importance

        # Normalize the feature importances (dividing by look_back as per your requirement)
        grouped_importances = {k: v / look_back for k, v in grouped_importances.items()}

        # Sort the importances in descending order
        sorted_importances = sorted(grouped_importances.items(), key=lambda x: x[1], reverse=True)

        # Print the sorted feature importances
        print("Feature Importances (Normalized):")
        for feature_name, importance in sorted_importances:
            print(f"{feature_name}: {importance:.4f}")


    pruning_methods = {
        "basic": prune_tree_basic,
        "average": prune_tree_average,
        "weighted": prune_tree_weighted,
    }

    if pruning_type not in pruning_methods:
        raise ValueError(f"Invalid pruning type '{pruning_type}'. Must be one of {list(pruning_methods.keys())}.")

    # Adjust the masked indices based on the look_back factor
    look_back = 10
    n_features = len(feature_names)
    adjusted_masked_indices = adjust_indices(masked_indices, look_back, n_features)

    # Prune and process the trees
    model_str, retained_trees = prune_and_process_trees(model, pruning_type, adjusted_masked_indices)

    #Re-boosting currently undegoing hardships
    #pruned_model = lgb.Booster(model_str=model_str)

    feature_importance = calculate_feature_importance_from_pruned_trees(retained_trees)
    #feature_importance = pruned_model.booster_.feature_importance(importance_type="gain")


    # Flatten test data
    if len(testX.shape) > 2:
        testX_reshaped = testX.reshape(testX.shape[0], -1)
    else:
        testX_reshaped = testX


    # rmse = root_mean_squared_error(testY, predictions)


    grouped_importances = {name: 0 for name in feature_names}
    for i, importance in enumerate(feature_importance):
        original_feature_idx = i % len(feature_names)
        original_feature_name = feature_names[original_feature_idx]
        grouped_importances[original_feature_name] += importance

    grouped_importances = {k: v / look_back for k, v in grouped_importances.items()}
    sorted_importances = sorted(grouped_importances.items(), key=lambda x: x[1], reverse=True)

    #return pruned_model, rmse, sorted_importances
    return None, None, sorted_importances
