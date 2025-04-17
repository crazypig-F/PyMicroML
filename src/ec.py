def aa_ec_map():
    # return {
    #     "serine": ['1.1.1.95', '2.6.1.52', '3.1.3.3'],
    #     "threonine": ['2.7.2.4', '1.2.1.11', '1.1.1.3', '2.7.1.39', '4.2.3.1'],
    #     "cysteine": ['2.5.1.47', '2.5.1.134', '4.4.1.1', '2.5.1.6', '4.2.1.22', '2.3.1.30', '3.2.2.9', '4.4.1.21'],
    #     "methionine": ['2.1.1.13', '2.3.1.46', '1.1.1.3', '1.2.1.11', '4.4.1.13', '2.7.2.4', '2.5.1.48'],
    #     "valine": ['2.2.1.6', '1.1.1.86', '4.2.1.9', '2.6.1.42'],
    #     "isoleucine": ['4.2.1.9', '1.1.1.85', '2.3.3.21', '2.2.1.6', '1.1.1.86', '2.6.1.42', '4.2.1.33',
    #                    '4.3.1.19'],
    #     "leucine": ['2.3.3.13', '4.2.1.33', '1.1.1.85'],
    #     "lysine": ['3.5.1.47', '5.1.1.7', '1.1.1.87', '3.5.1.18', '2.7.2.4', '2.7.2.17', '2.6.1.39', '2.6.1.17',
    #                '4.1.1.20', '1.17.1.8', '1.4.1.16', '1.5.1.10', '2.3.1.117', '4.3.3.7', '4.2.1.36', '2.6.1.118',
    #                '1.5.1.7', '3.5.1.130', '2.3.3.14', '2.3.1.89', '1.2.1.11', '6.3.2.43', '1.2.1.95', '1.2.1.103',
    #                '2.6.1.83'],
    #     "arginine": ['1.2.1.38', '2.1.3.3', '2.3.1.1', '3.5.1.16', '6.3.4.5', '2.6.1.11', '4.3.2.1', '2.1.3.9'],
    #     "proline": ['1.5.1.2', '1.2.1.41', '2.7.2.11'],
    #     "histidine": ['4.3.2.10', '2.6.1.9', '3.1.3.15', '5.3.1.16', '1.1.1.23', '4.2.1.19', '2.4.2.17', '3.6.1.31',
    #                   '3.5.4.19'],
    #     "tryptophan": ['4.1.3.27', '2.4.2.18', '5.3.1.24', '4.1.1.48', '4.2.1.20'],
    #     "phenylalanine": ['4.2.1.91', '5.4.99.5', '2.6.1.57', '2.6.1.78', '4.2.1.51'],
    #     "tyrosine": ['1.3.1.13', '5.4.99.5', '1.3.1.43', '2.6.1.57', '1.3.1.78', '2.6.1.78', '1.3.1.12'],
    #     "glycine": ['2.1.2.1', '4.1.2.48', '4.1.2.5', '2.6.1.44', '1.8.1.4', '2.1.2.10'],
    #     "alanine": ['2.6.1.2', '2.6.1.42', '2.8.1.7', '2.6.1.66'],
    #     "asparagine": ['6.1.1.23', '3.5.1.38', '3.5.1.2', '6.3.5.4', '6.3.1.1'],
    #     "glutamine": ['6.3.1.2'],
    #     "aspartate": ['2.6.1.1'],
    #     "glutamate": ['3.5,1.2', '3.5.1.38', '1.4.7.1', '1.2.1.88', '3.5.3.1', '1.4.1.14', '1.4.1.4', '1.4.1.3',
    #                   '2.6.1.13']
    # }
    return {
        "Histidine metabolism": ["2.4.2.17", "3.6.1.31", "3.5.4.19", "5.3.1.16", "4.3.2.10", "4.2.1.19", "2.6.1.9",
                                 "3.1.3.15", "4.2.1.19", "3.1.3.15", "1.1.1.23"],
        "Aromatic amino acid metabolism": ['2.5.1.54', '4.2.3.4', '4.2.1.10', '1.1.1.25', '2.7.1.71', '2.5.1.19',
                                           '4.2.3.5', '4.1.3.27', '2.4.2.18', '5.3.1.24', '4.1.1.48', '4.2.1.20',
                                           '5.4.99.5', '4.2.1.51', '2.6.1.57', '2.6.1.78', '2.6.1.79', '4.2.1.91',
                                           '1.3.1.12', '1.3.1.13', '1.3.1.43', '1.3.1.78'],
        "Branched-chain amino acid metabolism": ['2.2.1.6', '1.1.1.86', '4.2.1.9', '2.6.1.42', '2.3.3.21', '4.2.1.33',
                                                 '4.2.1.35', '1.1.1.85', '4.3.1.19', '2.3.3.13'],
        "Cysteine and methionine metabolism": ['2.3.1.30', '2.5.1.47', '4.2.1.22', '4.4.1.1', '2.5.1.6', '2.1.1.-',
                                               '3.2.2.9', '4.4.1.21', '2.5.1.134', '2.7.2.4', '1.2.1.11', '1.1.1.3',
                                               '2.3.1.46', '2.5.1.48', '4.4.1.13', '2.1.1.13', '2.1.1.14'],
        "Lysine metabolism": ['2.7.2.4', '1.2.1.11', '4.3.3.7', '1.17.1.8', '2.3.1.117', '2.6.1.17', '3.5.1.18',
                              '5.1.1.7', '4.1.1.20', '2.3.1.89', '2.6.1.-', '3.5.1.47', '1.4.1.16', '2.6.1.83',
                              '2.3.3.14', '4.2.1.-', '4.2.1.36', '1.1.1.87', '2.6.1.39', '1.2.1.95', '1.5.1.10',
                              '1.5.1.7', '4.2.1.114', '6.3.2.43', '2.7.2.17', '1.2.1.103', '2.6.1.118', '3.5.1.130'],
        "Arginine and proline metabolism": ['2.3.1.1', '2.7.2.8', '1.2.1.38', '2.6.1.11', '3.5.1.16', '3.5.1.14',
                                            '2.3.1.35', '6.3.2.-', '2.7.2.19', '1.2.1.-', '2.6.1.124', '3.5.1.132',
                                            '2.1.3.3', '6.3.4.5', '4.3.2.1', '2.1.3.9', '6.3.4.16', '3.5.3.1',
                                            '2.7.2.11', '1.2.1.41', '1.5.1.2'],
        "Other amino acid metabolism": ['1.1.1.95', '2.6.1.52', '4.1.2.5', '4.1.2.48', '4.2.3.1', '2.7.1.39', '2.6.1.2',
                                        '4.1.1.12', '2.6.1.1', '6.3.1.1', '6.3.5.4', '1.1.1.42', '1.1.1.41',
                                        '1.1.1.286', '1.4.1.13', '6.3.1.2']
    }
