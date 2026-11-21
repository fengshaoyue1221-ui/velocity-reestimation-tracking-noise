"""
Inventory of the trained checkpoints and the software environment (nothing is moved).
Usage (from MTR/, env mtr):  python ../missing_tools/inventory.py
Output: missing_outputs/checkpoints_and_environment.json
"""
import glob, hashlib, json, os, platform, subprocess, sys

def md5(path, chunk=1 << 22):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''): h.update(b)
    return h.hexdigest()

out = {'checkpoints': {}, 'environment': {}}
for p in sorted(glob.glob('output/waymo/*/*/ckpt/checkpoint_epoch_30.pth')):
    out['checkpoints'][p] = {'size_MB': round(os.path.getsize(p) / 2**20, 1), 'md5': md5(p)}
    print(p, out['checkpoints'][p])
env = out['environment']
env['python'] = sys.version.split()[0]; env['platform'] = platform.platform()
for mod in ['torch', 'tensorflow', 'numpy', 'waymo_open_dataset']:
    try:
        m = __import__(mod); env[mod] = getattr(m, '__version__', 'installed')
    except Exception as e:
        env[mod] = f'not importable: {e.__class__.__name__}'
try:
    import torch; env['cuda_available'] = torch.cuda.is_available(); env['gpus'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
except Exception:
    pass
try:
    env['nvidia_smi'] = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'],
                                       capture_output=True, text=True).stdout.strip().splitlines()
    env['pip_freeze'] = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True).stdout.strip().splitlines()
except Exception:
    pass
os.makedirs('../missing_outputs', exist_ok=True)
json.dump(out, open('../missing_outputs/checkpoints_and_environment.json', 'w'), indent=1)
print('saved ../missing_outputs/checkpoints_and_environment.json')
