import os
import subprocess as subp

from mendeleev import element
from scipy.optimize import differential_evolution as de


def which(program):
    import os

    def is_exe(exec_path):
        return os.path.isfile(exec_path) and os.access(exec_path, os.X_OK)

    file_path, file_name = os.path.split(program)
    if file_path:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None


def create_orca_input(symbols, coordinates, charge, mult, keywords,
                      extra_keywords, nprocs):
    name = "molecule"

    with open(f'{name}.inp', "w+") as file:
        file.write(f"{keywords}\n")
        if nprocs:
            # noinspection SpellCheckingInspection
            file.write(f"%pal nprocs {nprocs} end\n")
        if extra_keywords:
            file.write(f"{extra_keywords}\n")
        file.write(f"* xyz {charge} {mult}\n")
        for z, c in zip(symbols, coordinates):
            coordinate_line = f"{z:2s}  " \
                              f"{c[0]:12.8f}  " \
                              f"{c[1]:12.8f}  " \
                              f"{c[2]:12.8f}\n"
            file.write(coordinate_line)
        file.write('*\n')

        file.write("\n")
    return name


def create_gaussian_input(symbols, coordinates, charge, mult, keywords,
                          extra_keywords, nprocs):
    name = "molecule"

    with open(f'{name}.gjf', "w+") as file:
        file.write("%nosave\n")
        if nprocs:
            file.write(f"%nprocs={nprocs}\n")
        if extra_keywords:
            file.write(f"{extra_keywords}\n")
        file.write(f"{keywords}\n\n")
        file.write(f"{''.join(symbols)} {len(coordinates)} cluster\n\n")
        file.write(str(charge) + " " + str(mult) + "\n")
        for z, c in zip(symbols, coordinates):
            coordinate_line = f"{z:2s}  " \
                              f"{c[0]:12.8f}  " \
                              f"{c[1]:12.8f}  " \
                              f"{c[2]:12.8f}\n"
            file.write(coordinate_line)

        file.write("\n")
    return name


def read_gaussian_energy(name):
    file_to_open = f'{name}.log'
    with open(file_to_open, 'r+') as fp:
        return next(
            (
                float(line.split()[4])
                for line in fp.readlines()[::-1]
                if 'SCF Done' in line
            ),
            1e10,
        )


def read_orca_energy(name):
    file_to_open = f'{name}.out'
    with open(file_to_open, 'r+') as fp:
        return next(
            (
                float(line.split()[4])
                for line in fp.readlines()[::-1]
                if "FINAL SINGLE POINT ENERGY" in line
            ),
            1e10,
        )


def run_gaussian(name):
    with open(f'{name}.log', 'w') as out_n_err:
        out = subp.Popen(["g16", f'{name}.gjf'], stdout=out_n_err,
                         stderr=out_n_err)
        out.communicate()
        out.poll()
        exit_status = out.returncode
    return exit_status


def run_orca(name):
    # noinspection SpellCheckingInspection
    exe = which('orca')
    with open(f'{name}.out', 'w') as out_n_err:
        out = subp.Popen([exe, f'{name}.inp'], stdout=out_n_err,
                         stderr=out_n_err)
        out.communicate()
        out.poll()
        exit_status = out.returncode
    return exit_status


def write_xyz(coordinates, symbols, name, xyz_file_name):
    xyz_coordinate = coordinates.reshape((-1, 3))
    with open(xyz_file_name, 'a+') as fp:
        fp.writelines(f"{len(xyz_coordinate)}\n")
        fp.writelines(f"{name}\n")
        for symbol, line in zip(symbols, xyz_coordinate):
            fp.writelines(
                f"{symbol:2s} "
                f"{line[0]:12.8f} "
                f"{line[1]:12.8f} "
                f"{line[2]:12.8f}\n")


def calculate_g16_energy(coordinates, atoms, charge, multiplicity, keywords,
                         extra_keywords, nprocs):
    coordinate_i = coordinates.reshape((-1, 3))
    name = create_gaussian_input(atoms, coordinate_i, charge, multiplicity,
                                 keywords, extra_keywords, nprocs)
    exit_status = run_gaussian(name)
    return read_gaussian_energy(name) if exit_status == 0 else 1e10


def calculate_orca_energy(coordinates, atoms, charge, multiplicity, keywords,
                          extra_keywords, nprocs):
    coordinate_i = coordinates.reshape((-1, 3))
    name = create_orca_input(atoms, coordinate_i, charge, multiplicity,
                             keywords, extra_keywords, nprocs)
    exit_status = run_orca(name)
    return read_orca_energy(name) if exit_status == 0 else 1e10


def run_de(cla):
    atoms = cla['atoms']
    number_of_atoms = len(atoms)
    dim = number_of_atoms * 3
    bounds_factor = sum(element(z).atomic_radius for z in atoms) / 100.0
    lb = -bounds_factor / 2
    ub = bounds_factor / 2
    bounds = [(lb, ub)] * dim

    software = cla['software']
    cla.pop('software')
    if software == 'orca':
        objective_function = calculate_orca_energy
    elif software == 'gaussian':
        objective_function = calculate_g16_energy
    else:
        objective_function = None

    number_of_iterations = cla['n_iterations']
    cla.pop('n_iterations')
    arguments = (cla['atoms'], cla['charge'], cla["multiplicity"],
                 cla["keywords"], cla["extra_keywords"], cla["nprocs"])
    if os.path.exists('best_trj.xyz'):
        os.remove('best_trj.xyz')

    def coordinate_update(x, convergence):
        write_xyz(x, cla['atoms'], f"Convergence: {convergence}",
                  'best_trj.xyz')

    result = de(objective_function, bounds, args=arguments,
                polish=True, disp=True, callback=coordinate_update)
    print(result.message)
    final_energy = result.fun
    final_coordinates = result.x
    write_xyz(final_coordinates, atoms, f"Energy: {final_energy}", 'best.xyz')
    print("Global Best")
    print(f"Energy: {final_energy}")
    print(f"Coordinates\n{final_coordinates.reshape(-1, 3)}")
    return final_energy, final_coordinates


def main():
    import argparse
    program_description = """A program to do global optimization with 
    Differential Evolution algorithm using SciPy.optimize.differential_evolution
    Currently interfaced with ORCA and Gaussian packages.
    """
    example = """
    Example:
        python pyar-de.py C H H H H --software orca -c 0 -m 1 --keywords '! hf-3c'

        python pyar-de.py C H H H H --software gaussian -c 0 -m 1 --keywords '# b3lyp def2SVP'

    """
    parser = argparse.ArgumentParser(prog='pyar-de',
                                     description=program_description,
                                     epilog=example,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("atoms", metavar='Z',
                        type=str, nargs='+',
                        help='Atomic atoms of the atomic cluster. '
                             'For example, for a C4 cluster, provide C C C C. '
                             'For CH4, C H H H H')
    parser.add_argument("--software", type=str,
                        choices=['gaussian', 'orca'],
                        required=True, help="Software")
    # noinspection SpellCheckingInspection
    parser.add_argument('-nprocs', '--nprocs', metavar='n',
                        type=int, help='The number of processors/cores to be '
                                       'used by the quantum chemistry software.'
                        )
    parser.add_argument("-c", "--charge", type=int, required=True,
                        metavar='c',
                        help="Total charge of the system")
    parser.add_argument("-m", "--multiplicity", type=int,
                        required=True,
                        metavar='m',
                        help="Multiplicity of the system")
    parser.add_argument('--keywords', type=str, required=True,
                        help='Keyword line for the QM software input. '
                             'E.g., For gaussian input, "# PBE def2SVP"'
                             'Fro ORCA input, "! RI PBE def2-SVP D3BJ"')

    parser.add_argument('--extra-keywords', type=str,
                        help='Keyword line for the QM software input. '
                             'E.g., For gaussian input, "%%mem=2Gnk"'
                             'Fro ORCA input, and "%%scf maxiter=100 end"')

    parser.add_argument('--n-iterations', metavar='n', default=30,
                        type=int, help='The number of iterations. Default=30'
                        )
    args = parser.parse_args()
    energy, coordinate = run_de(vars(args))


if __name__ == "__main__":
    main()
