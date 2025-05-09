import re
import yaml

# === Step 1: Parse args.sh ===
def parse_args_sh(file_path):
    variables = {}
    args = {}
    current_group = None

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue # Skip comments and empty lines 
            line = re.sub(r'#.*', '', line).strip() # Remove inline comments
            if '=' in line and not line.endswith('('):
                key, val = line.split('=', 1)
                variables[key.strip()] = val.strip().strip('"')
            elif line.endswith('('):
                current_group = line[:-1].strip()
                args[current_group] = []
            elif line == ')':
                current_group = None
            elif current_group:
                args[current_group].append(line.strip().strip('"'))
    return variables, args

# === Step 2: Load the YAML config ===
def load_yaml_config(file_path):
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def compare_values(cli_val, yaml_val):
    try:
        return float(cli_val) == float(yaml_val)
    except (ValueError, TypeError):
        return str(cli_val) == str(yaml_val)

# === Step 3: Extract comparable fields ===
def extract_comparable_fields(parsed_args, parsed_yaml):
    mismatches = []

    # Compare hidden size
    cli_hidden_size = extract_value_from_args(parsed_args, '--hidden-size')
    yaml_hidden_size = parsed_yaml['model']['model_config']['hidden_size']
    if int(cli_hidden_size) != int(yaml_hidden_size):
        mismatches.append(('hidden-size', cli_hidden_size, yaml_hidden_size))

    # Add more key comparisons here
    pairs = [
        ('--num-layers', parsed_yaml['model']['model_config']['num_hidden_layers']),
        ('--num-attention-heads', parsed_yaml['model']['model_config']['num_attention_heads']),
        ('--ffn-hidden-size', parsed_yaml['model']['model_config']['intermediate_size']),
        ('--seq-length', parsed_yaml['tokens']['sequence_length']),
        ('--micro-batch-size', parsed_yaml['tokens']['micro_batch_size']),
        ('--global-batch-size', parsed_yaml['tokens']['micro_batch_size'] * parsed_yaml['tokens']['batch_accumulation_per_replica']),
        
        ('--norm-epsilon', parsed_yaml['model']['model_config']['rms_norm_eps']),
        ('--num-query-groups', parsed_yaml['model']['model_config']['num_attention_heads'] // parsed_yaml['model']['model_config']['num_key_value_heads']),
        ('--rotary-base', parsed_yaml['model']['model_config']['rope_theta']),

        # Optimizer
        ('--lr', parsed_yaml['optimizer']['learning_rate_scheduler']['learning_rate']),
        ('--min-lr', parsed_yaml['optimizer']['learning_rate_scheduler']['min_decay_lr']),
        ('--weight-decay', parsed_yaml['optimizer']['weight_decay']),
        ('--adam-beta1', parsed_yaml['optimizer']['optimizer_factory']['adam_beta1']),
        ('--adam-beta2', parsed_yaml['optimizer']['optimizer_factory']['adam_beta2']),
        ('--adam-eps', parsed_yaml['optimizer']['optimizer_factory']['adam_eps']),
        ('--lr-decay-style', parsed_yaml['optimizer']['learning_rate_scheduler']['lr_decay_style']),
        # need to check this line
        ('--lr-decay-iters', parsed_yaml['optimizer']['learning_rate_scheduler']['lr_decay_steps'] + parsed_yaml['optimizer']['learning_rate_scheduler']['lr_warmup_steps']),
        ('--lr-warmup-iters', parsed_yaml['optimizer']['learning_rate_scheduler']['lr_warmup_steps']),

        # MoE
        ('--moe-aux-loss-coeff', parsed_yaml['model']['model_config']['moe_config']['aux_loss_coeff']) if 'aux_loss_coeff' in parsed_yaml['model']['model_config']['moe_config'] else (None, None),
        ('--moe-router-load-balancing-type', parsed_yaml['model']['model_config']['moe_config']['load_balancing_type']) if 'load_balancing_type' in parsed_yaml['model']['model_config']['moe_config'] else (None, None),
        ('--moe-router-topk', parsed_yaml['model']['model_config']['moe_config']['top_k']),
        ('--moe-shared-expert-intermediate-size', parsed_yaml['model']['model_config']['moe_config']['shared_expert_intermediate_size']),
        ('--moe-ffn-hidden-size', parsed_yaml['model']['model_config']['moe_config']['moe_intermediate_size']),
        ('--moe-token-dispatcher-type', parsed_yaml['model']['model_config']['moe_config']['token_dispatcher_type']),
    ]
        


    for cli_key, yaml_val in pairs:
        if cli_key == None and yaml_val == None:
            continue
        cli_val = extract_value_from_args(parsed_args, cli_key)
        if cli_val is None:
            raise ValueError(f"Warning: {cli_key} not found in args.sh")
        if not compare_values(cli_val, yaml_val):
            mismatches.append((cli_key, cli_val, yaml_val))

    return mismatches

def extract_value_from_args(parsed_args, key):
    for group_args in parsed_args.values():
        for arg in group_args:
            if arg.startswith(key):
                parts = arg.split()
                if len(parts) > 1:
                    return parts[1]
    return None

def assert_required_flags_exist(parsed_args, required_flags):
    all_args = [arg for group in parsed_args.values() for arg in group]
    missing_flags = [flag for flag in required_flags if not any(arg.startswith(flag) for arg in all_args)]
    
    if missing_flags:
        raise ValueError(f"Missing required flags in args.sh: {missing_flags}")

required_flags = [
    '--moe-router-pre-softmax',
]

# === Step 4: Main ===
def compare_configs(sh_file_path, yaml_file_path):
    variables, parsed_args = parse_args_sh(sh_file_path)
    parsed_yaml = load_yaml_config(yaml_file_path)
    assert_required_flags_exist(parsed_args, required_flags)
    mismatches = extract_comparable_fields(parsed_args, parsed_yaml)

    # Assert optimizer name is adam
    optimizer_name = parsed_yaml.get('optimizer', {}).get('optimizer_factory', {}).get('name', None)
    if optimizer_name != 'adam':
        raise ValueError(f"Expected optimizer name to be 'adam', but got '{optimizer_name}'")
    
    if mismatches:
        for key, cli_val, yaml_val in mismatches:
            print(f"Mismatch for {key}: args.sh='{cli_val}' vs yaml='{yaml_val}'")
        raise ValueError("Config mismatch found.")
    else:
        print("success")

# === Usage ===

megatron_config = '/fsx/haojun/Megatron-files/config/qwen_moe/moe_250m_aux_loss_long.sh'
# nanotron_config = '/fsx/haojun/training_scripts/config/qwen/megatron/qwen_225M_aux_loss_long.yaml'
nanotron_config = '/fsx/haojun/training_scripts/config/qwen/megatron/qwen_225M_long.yaml'

compare_configs(megatron_config, nanotron_config)

