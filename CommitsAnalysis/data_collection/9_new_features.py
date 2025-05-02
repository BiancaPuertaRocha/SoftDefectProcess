import pandas as pd
import os
import sys

def extract_patch_features(df):
    def process_patch(patch):
        if pd.isna(patch):
            return {
                'patch_size': 0,
                'patch_has_new_test': 0,
                'patch_num_control_flow_changes': 0,
                'patch_comment_lines_added': 0,
                'patch_max_added_line_length': 0,
                'patch_has_exception_handling_change': 0,
            }

        lines = patch.split('\n')
        added_lines = [line for line in lines if line.startswith('+') and not line.startswith('+++')]

        has_new_test = any(('test' in line.lower() or 'assert' in line.lower()) for line in added_lines)

        control_flow_keywords = ['if', 'for', 'while', 'switch', 'case', 'else']
        num_control_flow_changes = sum(any(keyword in line for keyword in control_flow_keywords) for line in added_lines)

        comment_patterns = ['//', '#', '/*', '*', '--']
        comment_lines_added = sum(any(pattern in line for pattern in comment_patterns) for line in added_lines)

        max_added_line_length = max((len(line) for line in added_lines), default=0)

        exception_keywords = ['try', 'catch', 'throw', 'throws', 'except']
        has_exception_handling_change = any(any(keyword in line for keyword in exception_keywords) for line in added_lines)

        return {
            'patch_size': len(lines),
            'patch_has_new_test': int(has_new_test),
            'patch_num_control_flow_changes': num_control_flow_changes,
            'patch_comment_lines_added': comment_lines_added,
            'patch_max_added_line_length': max_added_line_length,
            'patch_has_exception_handling_change': int(has_exception_handling_change),
        }

    patch_features = df['patch'].apply(process_patch)
    patch_features_df = pd.DataFrame(list(patch_features))

    total_changes = df['additions'].fillna(0) + df['deletions'].fillna(0)
    df['patch_additions_ratio'] = df['additions'].fillna(0) / total_changes.replace(0, 1)
    df['patch_deletions_ratio'] = df['deletions'].fillna(0) / total_changes.replace(0, 1)

    df = pd.concat([df.reset_index(drop=True), patch_features_df.reset_index(drop=True)], axis=1)

    return df

def main():
    input_path = sys.argv[1]

    # Carrega o CSV
    df = pd.read_csv(input_path)

    # Aplica extração de features
    df = extract_patch_features(df)

    # Gera o novo caminho para salvar
    dir_name = os.path.dirname(input_path)
    base_name = os.path.basename(input_path)
    name, ext = os.path.splitext(base_name)
    output_name = f"{name}_new_features{ext}"
    output_path = os.path.join(dir_name, output_name)

    # Salva o novo CSV
    df.to_csv(output_path, index=False)

    print(f"Arquivo salvo em: {output_path}")

if __name__ == "__main__":
    main()
