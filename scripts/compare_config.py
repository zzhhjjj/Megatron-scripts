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
        ('--global-batch-size', parsed_yaml['tokens']['micro_batch_size'] * parsed_yaml['tokens']['batch_accumulation_per_replica'] * parsed_yaml['parallelism']['dp']),
        ('--vocab-size', parsed_yaml['model']['model_config']['vocab_size']),
        
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

        # Parallelism
        ('--tensor-model-parallel-size', parsed_yaml['parallelism']['tp']),
        ('--pipeline-model-parallel-size', parsed_yaml['parallelism']['pp']),
    ]
        
    is_moe = 'moe_config' in parsed_yaml['model']
    if is_moe:
        pairs.append(('--moe-aux-loss-coeff', parsed_yaml['model']['model_config']['moe_config']['aux_loss_coeff']))
        pairs.append(('--moe-router-load-balancing-type', parsed_yaml['model']['model_config']['moe_config']['load_balancing_type']))
        pairs.append(('--moe-router-topk', parsed_yaml['model']['model_config']['moe_config']['top_k']))
        pairs.append(('--moe-shared-expert-intermediate-size', parsed_yaml['model']['model_config']['moe_config']['shared_expert_intermediate_size']))
        pairs.append(('--moe-ffn-hidden-size', parsed_yaml['model']['model_config']['moe_config']['moe_intermediate_size']))

    # Check required flags in Megatron args
    required_flags = [
        '--overlap-grad-reduce',
        '--disable-bias-linear',
        '--attention-dropout',
        '--hidden-dropout',
        '--swiglu',
    ]
    if is_moe:
        required_flags.append('--moe-router-pre-softmax')
    all_args = [arg for group in parsed_args.values() for arg in group]
    missing_flags = [flag for flag in required_flags if not any(arg.startswith(flag) for arg in all_args)]
    
    # Check required settings in YAML config
    required_yaml_settings = {
        'general.ignore_sanity_checks': True,
        'optimizer.accumulate_grad_in_fp32': True,
        'model.model_config.tie_word_embeddings': True,
        'model.model_config._fused_rms_norm': True,
        'model.model_config._fused_rotary_emb': True,
        'model.model_config._use_qkv_packed': True,
        'model.model_config.z_loss_enabled': False,
    }
    missing_yaml_settings = []
    for path, expected_value in required_yaml_settings.items():
        keys = path.split('.')
        current = parsed_yaml
        try:
            for key in keys:
                current = current[key]
            if current != expected_value:
                missing_yaml_settings.append(f"{path} should be {expected_value}, got {current}")
        except (KeyError, TypeError):
            missing_yaml_settings.append(f"{path} not found")
        
        if missing_yaml_settings:
            raise ValueError(f"Missing or incorrect required settings in YAML config: {missing_yaml_settings}")

    if missing_flags:
        raise ValueError(f"Missing required flags in args.sh: {missing_flags} in Megatron config")


    for cli_key, yaml_val in pairs:
        if cli_key == None and yaml_val == None:
            continue
        cli_val = extract_value_from_args(parsed_args, cli_key)
        if cli_val is None:
            raise ValueError(f"Warning: {cli_key} not found in Megatron config")
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
    

# === Step 4: Main ===
def compare_configs(sh_file_path, yaml_file_path):
    variables, parsed_args = parse_args_sh(sh_file_path)
    parsed_yaml = load_yaml_config(yaml_file_path)
    mismatches = extract_comparable_fields(parsed_args, parsed_yaml)
    
    print("Comparing config:")
    print(f"Megatron config: {sh_file_path}")
    print(f"Nanotron config: {yaml_file_path}")
    if mismatches:
        for key, cli_val, yaml_val in mismatches:
            print(f"Mismatch for {key}:\n  Megatron config='{cli_val}'\n  Nanotron config='{yaml_val}'")
        raise ValueError("Config mismatch found.")
    else:
        print("success")

# === Usage ===

# MoE config 
# megatron_config = '/fsx/haojun/Megatron-files/config/qwen_moe/moe_250m_aux_loss_long.sh'
# nanotron_config = '/fsx/haojun/training_scripts/config/qwen/megatron/qwen_225M_aux_loss_long.yaml'
# nanotron_config = '/fsx/haojun/training_scripts/config/qwen/megatron/qwen_225M_long.yaml'

# Dense config
megatron_config = '/fsx/haojun/Megatron-files/config/dense/megatron/dense_104M.sh'
nanotron_config = '/fsx/haojun/Megatron-files/config/dense/nanotron/dense_104M.yaml'
compare_configs(megatron_config, nanotron_config)

# 1B 
megatron_config = '/fsx/haojun/Megatron-files/config/dense/megatron/dense_1B.sh'
nanotron_config = '/fsx/haojun/Megatron-files/config/dense/nanotron/dense_1B.yaml'
compare_configs(megatron_config, nanotron_config)

# 8B
megatron_config = '/fsx/haojun/Megatron-files/config/dense/megatron/dense_8B.sh'
nanotron_config = '/fsx/haojun/Megatron-files/config/dense/nanotron/dense_8B.yaml'
compare_configs(megatron_config, nanotron_config)


