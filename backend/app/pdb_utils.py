import os
import requests
import random
from typing import List
from Bio import PDB
from Bio.PDB import PDBParser, DSSP
from Bio.SeqUtils import seq1
from motor.motor_asyncio import AsyncIOMotorClient

# MongoDB setup
MONGO_URI = "mongodb://mongo:27017"
client = AsyncIOMotorClient(MONGO_URI)
db = client["protein_db"]

# AA property definitions
AA_PROPERTIES = {
    'A': {'hydrophobicity': 1.8, 'volume': 88.6, 'charge': 0, 'polarity': 8.1, 'side_chain': 1},
    'R': {'hydrophobicity': -4.5, 'volume': 173.4, 'charge': 1, 'polarity': 10.5, 'side_chain': 1},
    'N': {'hydrophobicity': -3.5, 'volume': 114.1, 'charge': 0, 'polarity': 11.6, 'side_chain': 1},
    'D': {'hydrophobicity': -3.5, 'volume': 111.1, 'charge': -1, 'polarity': 13.0, 'side_chain': 1},
    'C': {'hydrophobicity': 2.5, 'volume': 108.5, 'charge': 0, 'polarity': 5.5, 'side_chain': 1},
    'Q': {'hydrophobicity': -3.5, 'volume': 143.8, 'charge': 0, 'polarity': 10.5, 'side_chain': 1},
    'E': {'hydrophobicity': -3.5, 'volume': 138.4, 'charge': -1, 'polarity': 12.3, 'side_chain': 1},
    'G': {'hydrophobicity': -0.4, 'volume': 60.1, 'charge': 0, 'polarity': 9.0, 'side_chain': 0},
    'H': {'hydrophobicity': -3.2, 'volume': 153.2, 'charge': 0.1, 'polarity': 10.4, 'side_chain': 1},
    'I': {'hydrophobicity': 4.5, 'volume': 166.7, 'charge': 0, 'polarity': 5.2, 'side_chain': 1},
    'L': {'hydrophobicity': 3.8, 'volume': 166.7, 'charge': 0, 'polarity': 4.9, 'side_chain': 1},
    'K': {'hydrophobicity': -3.9, 'volume': 168.6, 'charge': 1, 'polarity': 11.3, 'side_chain': 1},
    'M': {'hydrophobicity': 1.9, 'volume': 162.9, 'charge': 0, 'polarity': 5.7, 'side_chain': 1},
    'F': {'hydrophobicity': 2.8, 'volume': 189.9, 'charge': 0, 'polarity': 5.2, 'side_chain': 1},
    'P': {'hydrophobicity': -1.6, 'volume': 112.7, 'charge': 0, 'polarity': 8.0, 'side_chain': 1},
    'S': {'hydrophobicity': -0.8, 'volume': 89.0, 'charge': 0, 'polarity': 9.2, 'side_chain': 1},
    'T': {'hydrophobicity': -0.7, 'volume': 116.1, 'charge': 0, 'polarity': 8.6, 'side_chain': 1},
    'W': {'hydrophobicity': -0.9, 'volume': 227.8, 'charge': 0, 'polarity': 5.4, 'side_chain': 1},
    'Y': {'hydrophobicity': -1.3, 'volume': 193.6, 'charge': 0, 'polarity': 6.2, 'side_chain': 1},
    'V': {'hydrophobicity': 4.2, 'volume': 140.0, 'charge': 0, 'polarity': 5.9, 'side_chain': 1}
}

AA_GROUPS = {
    'nonpolar': set('AVILMFYW'),
    'polar': set('STNQ'),
    'charged': set('DEKRH'),
    'aromatic': set('FYW'),
}

def aa_group_flags(aa: str):
    return {
        'nonpolar': int(aa in AA_GROUPS['nonpolar']),
        'polar': int(aa in AA_GROUPS['polar']),
        'charged': int(aa in AA_GROUPS['charged']),
        'aromatic': int(aa in AA_GROUPS['aromatic'])
    }

def get_features_for_residue(residue, dssp_dict, chain_id):
    aa = residue.resname
    if len(aa) != 3:
        return None
    try:
        aa_short = seq1(aa)
    except Exception:
        return None
    props = AA_PROPERTIES.get(aa_short)
    key = (chain_id, residue.id[1])
    if props is None or key not in dssp_dict:
        return None
    return {
        'aa': aa_short,
        **props,
        **aa_group_flags(aa_short)
    }

async def fetch_and_store_protein(pdb_id: str, n_before: int = 3, n_inside: int = 4):
    entry = await db.proteins.find_one({"pdb_id": pdb_id})
    
    # If everything already exists
    if entry and "metadata" in entry and "pdb_stored" in entry and entry["pdb_stored"]:
        return {"status": "exists", "pdb_id": pdb_id, "n_rows": len(entry.get("features", []))}

    # Metadata retrieval (only if missing)
    if not entry or "metadata" not in entry:
        metadata_url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
        r = requests.get(metadata_url)
        if r.status_code != 200:
            return {"status": "error", "pdb_id": pdb_id}
        metadata = r.json()
    else:
        metadata = entry["metadata"]

    # Download PDB file
    pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    pdb_path = f"temp_{pdb_id}.pdb"
    with open(pdb_path, "w") as f:
        f.write(requests.get(pdb_url).text)

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, pdb_path)
    model = structure[0]

    try:
        dssp = DSSP(model, pdb_path)
    except Exception as e:
        os.remove(pdb_path)
        return {"status": "dssp_failed", "error": str(e), "pdb_id": pdb_id}

    dssp_dict = {(chain, res_id[1]): dssp[key] for key in dssp.keys() for chain, res_id in [key]}
    residues = [(res, chain.id) for chain in model for res in chain if res.id[0] == ' ']
    residue_ids = [(chain_id, res.id[1]) for res, chain_id in residues]

    feature_keys = ['hydrophobicity', 'polarity', 'volume', 'charge', 'side_chain', 'nonpolar', 'polar', 'charged', 'aromatic']
    rows = []

    for idx in range(n_before, len(residues) - n_inside):
        chain_id, res_id = residue_ids[idx]
        key = (chain_id, res_id)

        is_positive = (
            key in dssp_dict and dssp_dict[key][2] == 'H' and
            all(
                (residue_ids[idx + j][0], residue_ids[idx + j][1]) in dssp_dict and
                dssp_dict[(residue_ids[idx + j][0], residue_ids[idx + j][1])][2] == 'H'
                for j in range(n_inside)
            )
        )

        row = {}
        valid = True
        for offset in range(-n_before, n_inside):
            res_idx = idx + offset
            res, ch_id = residues[res_idx]
            feat = get_features_for_residue(res, dssp_dict, ch_id)
            if not feat:
                valid = False
                break
            for k, v in feat.items():
                row[f"{k}_{offset}"] = v
        if valid:
            row["label"] = int(is_positive)
            rows.append(row)

    os.remove(pdb_path)

    # Update or insert in database
    await db.proteins.update_one(
        {"pdb_id": pdb_id},
        {
            "$set": {
                "metadata": metadata,
                "pdb_stored": True,
                "features": rows
            }
        },
        upsert=True
    )

    return {"status": "stored", "pdb_id": pdb_id, "n_rows": len(rows)}
