import generator_AAPM_3d_sparseview

if __name__ == '__main__':
    #generator_AAPM_3d_sparseview.run(p_metho='TGV', p_n_view=8, p_rho_0=5, p_rho_1=5, p_alpha_0=2.5, p_alpha_1=2.5, p_lam=0.04)
    generator_AAPM_3d_sparseview.run(p_metho='TGV', p_n_view=4, p_rho_0=5, p_rho_1=5, p_alpha_0=2.5, p_alpha_1=2.5, p_lam=0.04)
    generator_AAPM_3d_sparseview.run(p_metho='TGV', p_n_view=2, p_rho_0=5, p_rho_1=5, p_alpha_0=2.5, p_alpha_1=2.5, p_lam=0.04)

    generator_AAPM_3d_sparseview.run(p_metho='TV', p_n_view=8, p_rho_0=10, p_lam=0.04)
    generator_AAPM_3d_sparseview.run(p_metho='TV', p_n_view=4, p_rho_0=10, p_lam=0.04)
    generator_AAPM_3d_sparseview.run(p_metho='TV', p_n_view=2, p_rho_0=10, p_lam=0.04)