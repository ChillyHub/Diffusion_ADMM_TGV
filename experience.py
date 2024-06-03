import argparse

import generator_AAPM_3d_sparseview
import generator_AAPM_3d_limitedangle

if __name__ == '__main__':
    #generator_AAPM_3d_sparseview.run(p_metho='TGV', p_n_view=8, p_rho_0=5, p_rho_1=5, p_alpha_0=2.5, p_alpha_1=2.5, p_lam=0.04)
    #generator_AAPM_3d_sparseview.run(p_metho='TGV', p_n_view=4, p_rho_0=5, p_rho_1=5, p_alpha_0=2.5, p_alpha_1=2.5, p_lam=0.04)
    #generator_AAPM_3d_sparseview.run(p_metho='TGV', p_n_view=2, p_rho_0=5, p_rho_1=5, p_alpha_0=2.5, p_alpha_1=2.5, p_lam=0.04)

    #generator_AAPM_3d_sparseview.run(p_metho='TGV_old', p_n_view=8, p_rho_0=5, p_rho_1=5, p_alpha_0=2.5, p_alpha_1=2.5, p_lam=0.04)
    #generator_AAPM_3d_sparseview.run(p_metho='TGV_old', p_n_view=4, p_rho_0=5, p_rho_1=5, p_alpha_0=2.5, p_alpha_1=2.5, p_lam=0.04)
    #generator_AAPM_3d_sparseview.run(p_metho='TGV_old', p_n_view=2, p_rho_0=5, p_rho_1=5, p_alpha_0=2.5, p_alpha_1=2.5, p_lam=0.04)

    #generator_AAPM_3d_sparseview.run(p_metho='TV', p_n_view=8, p_rho_0=10, p_lam=0.04)
    #generator_AAPM_3d_sparseview.run(p_metho='TV', p_n_view=4, p_rho_0=10, p_lam=0.04)
    #generator_AAPM_3d_sparseview.run(p_metho='TV', p_n_view=2, p_rho_0=10, p_lam=0.04)

    parser = argparse.ArgumentParser(description='Experience.')

    parser.add_argument('--view', type=int, default=0, help='Number of views')
    parser.add_argument('--method', type=str, default='', help='Method')
    args = parser.parse_args()

    methods = args.method.split(',')

    if args.view == 8 or args.view == 0:
        if 'TGV' in methods or args.method == '':
            generator_AAPM_3d_sparseview.run(p_metho='TGV', p_n_view=8, p_rho_0=5, p_rho_1=5, p_alpha_0=2.5, p_alpha_1=2.5, p_lam=0.04)
        if 'TGV_old' in methods or args.method == '':
            generator_AAPM_3d_sparseview.run(p_metho='TGV_old', p_n_view=8, p_rho_0=5, p_rho_1=5, p_alpha_0=2.5, p_alpha_1=2.5, p_lam=0.04)
        if 'TV' in methods or args.method == '':
            generator_AAPM_3d_sparseview.run(p_metho='TV', p_n_view=8, p_rho_0=10, p_lam=0.04)

    if args.view == 4 or args.view == 0:
        if 'TGV' in methods or args.method == '':
            generator_AAPM_3d_sparseview.run(p_metho='TGV', p_n_view=4, p_rho_0=5, p_rho_1=5, p_alpha_0=2.5, p_alpha_1=2.5, p_lam=0.04)
        if 'TGV_old' in methods or args.method == '':
            generator_AAPM_3d_sparseview.run(p_metho='TGV_old', p_n_view=4, p_rho_0=5, p_rho_1=5, p_alpha_0=2.5, p_alpha_1=2.5, p_lam=0.04)
        if 'TV' in methods or args.method == '':
            generator_AAPM_3d_sparseview.run(p_metho='TV', p_n_view=4, p_rho_0=10, p_lam=0.04)

    if args.view == 2 or args.view == 0:
        if 'TGV' in methods or args.method == '':
            generator_AAPM_3d_sparseview.run(p_metho='TGV', p_n_view=2, p_rho_0=5, p_rho_1=5, p_alpha_0=2.5, p_alpha_1=2.5, p_lam=0.04)
        if 'TGV_old' in methods or args.method == '':
            generator_AAPM_3d_sparseview.run(p_metho='TGV_old', p_n_view=2, p_rho_0=5, p_rho_1=5, p_alpha_0=2.5, p_alpha_1=2.5, p_lam=0.04)
        if 'TV' in methods or args.method == '':
            generator_AAPM_3d_sparseview.run(p_metho='TV', p_n_view=2, p_rho_0=10, p_lam=0.04)

    if args.view == 90 or args.view == 0:
        if 'TGV' in methods or args.method == '':
            generator_AAPM_3d_limitedangle.run(p_metho='TGV', p_n_view=90, p_rho_0=5, p_rho_1=5, p_alpha_0=2.5, p_alpha_1=2.5, p_lam=0.04)
        if 'TGV_old' in methods or args.method == '':
            generator_AAPM_3d_limitedangle.run(p_metho='TGV_old', p_n_view=90, p_rho_0=5, p_rho_1=5, p_alpha_0=2.5, p_alpha_1=2.5, p_lam=0.04)
        if 'TV' in methods or args.method == '':
            generator_AAPM_3d_limitedangle.run(p_metho='TV', p_n_view=90, p_rho_0=10, p_lam=0.04)
            