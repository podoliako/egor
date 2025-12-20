import numpy as np
import subprocess
import tempfile
import os


os.environ["LD_LIBRARY_PATH"] = ":".join([
    "/mnt/disk01/egor/ucvm_final/lib",
    "/mnt/disk01/egor/ucvm_final/lib/proj/lib",
    os.environ.get("LD_LIBRARY_PATH", "")
])


def get_velocities(depths, lons, lats, cube_side):

    # Чтобы зацепить станции у поверхности
    for i in range(len(depths)):
        if -depths[i] >= 0 and -depths[i] <= cube_side:
            depths[i] = 0

    # print(depths)
            
    vs_values = np.zeros_like(lons)
    vp_values = np.zeros_like(lons)
    
    with tempfile.NamedTemporaryFile(mode='w+') as f:
        for lon, lat, depth in zip(np.array(lons).ravel(), np.array(lats).ravel(), np.array(depths).ravel()):
            f.write(f"{lon} {lat} {depth}\n")
        f.flush()
        
        cmd = [
            "/mnt/disk01/egor/ucvm_final/bin/ucvm_query",
            "-f", "/mnt/disk01/egor/ucvm_final/conf/ucvm.conf",
            "-m", "cvmh",
            "<", f.name
        ]
        
        result = subprocess.run(
            ' '.join(cmd),
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        for i, line in enumerate(result.stdout.splitlines()):
            parts = line.split()
            vp_values.flat[i] = float(parts[6])
            vs_values.flat[i] = float(parts[7])
            
    return vp_values.reshape(len(depths)), vs_values.reshape(len(depths))
