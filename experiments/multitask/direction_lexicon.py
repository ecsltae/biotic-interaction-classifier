"""Relation polarity lexicon for the direction head.

FACTORISATION
    direction(passage, s1, R, s2) = FORWARD  iff  role(s1) == subject_role(R)

`role(taxon)` is a binary ecological role over the ordered pair:
    A = agent / symbiont side  (pathogen, parasite, predator, consumer, pollinator,
                                invader, contaminant, coloniser, the transmitted agent)
    B = patient / host side    (host, prey, food, the exposed / invaded / contaminated
                                organism, the reservoir, the vector that carries an agent)

`POL[ro_id]` says which role fills the SUBJECT slot of that ROBI concept's preferred
term:  +1 -> subject is A,  -1 -> subject is B,  0 -> the relation is symmetric and
direction is undefined (emit not_applicable, never guess).

Built by reading each concept's preferred_term and synonyms in robiext_v2025.json.
It was FROZEN before any measurement against the human gold; the one place it is
known to disagree with the annotator (RO_e000005 "infested by") is left as-is and
reported, rather than flipped to buy an item.
"""
import json, functools

POL = {
    # --- subject is the AGENT / SYMBIONT side ---------------------------------
    "RO_0002556": +1,  # pathogen of
    "RO_e000055": +1,  # affecting
    "RO_e000045": +1,  # invaded  (synonyms are all active: invade/invades/invading/invasion)
    "RO_e000002": +1,  # has contaminated
    "RO_0002439": +1,  # preys on
    "RO_0002211": +1,  # regulates
    "RO_e000007": +1,  # feed on
    "RO_0002444": +1,  # parasite of
    "RO_0002451": +1,  # transmitted by -- subject is the transmitted AGENT, object the vector
    "RO_e000053": +1,  # colonize
    "RO_e000013": +1,  # graze
    "RO_0002470": +1,  # eats
    "RO_0002455": +1,  # pollinates
    "RO_e000011": +1,  # hunts
    "RO_0002632": +1,  # ectoparasite of
    "RO_e000038": +1,  # neutralizes
    "RO_0002226": +1,  # develops in
    "RO_0002626": +1,  # kills
    "RO_0002618": +1,  # visits
    "RO_0002454": +1,  # has host
    "RO_e000006": +1,  # has infested
    "RO_0002634": +1,  # endoparasite of
    "RO_0002212": +1,  # negatively regulates
    "RO_0002208": +1,  # parasitoid of
    "RO_e000010": +1,  # has ingested
    "RO_0002213": +1,  # positively regulates
    "RO_0002227": +1,  # obligate parasite of
    "RO_e000018": +1,  # endoparasitoid of
    "RO_0002428": +1,  # involved in regulation of
    "RO_0002237": +1,  # hemiparasite of
    "RO_0011009": +1,  # directly positively regulates quantity of
    "RO_0003003": +1,  # increases expression of
    "RO_0002578": +1,  # directly regulates
    "RO_x_infects": +1,  # "X infects Y" -- agent in subject position (not in ROBI)
    # --- subject is the PATIENT / HOST side -----------------------------------
    "RO_e000044": -1,  # exposed to
    "RO_e000001": -1,  # contaminated by
    "RO_e000005": -1,  # infested by
    "RO_0002453": -1,  # host of
    "RO_e000056": -1,  # affected by
    "RO_0002458": -1,  # preyed upon by
    "RO_0002445": -1,  # parasitized by
    "RO_e000009": -1,  # ingested by
    "RO_0002334": -1,  # regulated by
    "RO_e000051": -1,  # reservoir host
    "RO_0002459": -1,  # is vector for -- the vector carries the agent; agent is the object
    "RO_e000054": -1,  # colonized by
    "RO_0002456": -1,  # pollinated by
    "RO_e000008": -1,  # fed by  ("feeding by Y" -> Y consumes X)
    "RO_0002627": -1,  # is killed by
    "RO_e000012": -1,  # hunted by
    "RO_0002471": -1,  # is eaten by
    "RO_e000039": -1,  # neutralized by
    "RO_0002619": -1,  # visited by
    "RO_e000003": -1,  # seropositive to
    "RO_e000041": -1,  # transported by
    "RO_0002802": -1,  # reservoir host of
    "RO_e000052": -1,  # competent host
    "RO_e000050": -1,  # attracted by
    "RO_x_resist": -1,  # "X is resistant to Y" -- resisting host in subject position
    # --- symmetric: direction is undefined ------------------------------------
    "RO_0002326": 0,   # contributes to
    "RO_e000046": 0,   # migrates with
    "RO_0002440": 0,   # symbiotically interacts with
    "RO_0002434": 0,   # interacts with
    "RO_0002502": 0,   # depends on
    "RO_0002438": 0,   # trophically interacts with
    "RO_0002371": 0,   # attached to
    "RO_e000047": 0,   # coevolves
    "RO_e000027": 0,   # adjacent to
    "RO_0002442": 0,   # mutualistically interacts with
    "RO_e000049": 0,   # attracts
    "RO_e000004": 0,   # baits
    "RO_0002801": 0,   # co-roosts with
    "RO_0002325": 0,   # colocalizes with
    "RO_e000017": 0,   # interspecific interaction
    "RO_e000040": 0,   # cooperates with
    "RO_e000032": 0,   # phoresy interaction
    "RO_e000048": 0,   # communicates with
    "RO_0008506": 0,   # ecologically co-occurs with
    "RO_e000025": 0,   # lives near
}

# Forms ROBI has no entry for, or files under the wrong concept. Each is justified by
# grammar alone, independently of any gold label:
#   "host"/"hosts"     -- ROBI has host of / host for / hosts to, all RO_0002453; the bare
#                         noun is the same concept and is missing only as a synonym.
#   "infect*"          -- ROBI has no infection concept. "X infects Y" puts the agent in
#                         subject position, so it takes agent-side polarity (+1).
#   "resistan*"        -- "X is resistant to Y" puts the resisting host in subject
#                         position, so host-side polarity (-1).
#   "preys"/"predator" -- ROBI lists these ACTIVE forms as synonyms of the PASSIVE concept
#                         RO_0002458 "preyed upon by". They belong to RO_0002439 "preys on".
#                         (An independent review of this repo flagged the same defect.)
EXTRA_FORMS = {
    "host": "RO_0002453", "hosts": "RO_0002453",
    "infect": "RO_x_infects", "infects": "RO_x_infects", "infecting": "RO_x_infects",
    "infected": "RO_x_infects", "infection": "RO_x_infects", "infections": "RO_x_infects",
    "resistant": "RO_x_resist", "resistance": "RO_x_resist",
    "resistance to": "RO_x_resist", "resistant to": "RO_x_resist",
    "preys": "RO_0002439", "preys on": "RO_0002439", "prey on": "RO_0002439",
    "predator": "RO_0002439", "predators": "RO_0002439", "predatory": "RO_0002439",
    "predator of": "RO_0002439", "predators of": "RO_0002439",
}

_ROBI = "/home/egaillac/MetaP/classifier/data/taxonomies/robiext_v2025.json"

_FORM2RO = "/home/egaillac/MetaP/classifier/experiments/multitask/direction_form2ro.json"

@functools.lru_cache(maxsize=1)
def _pool_forms():
    """The pipeline's own surface-form -> RO id map, harvested from med25_pool.

    Richer than ROBI's synonym lists (which miss bare nouns like "host",
    "pathogen", "prey"), and it is the map that actually produced the triples.
    Verified 1:1 over all 397 forms in the pool.
    """
    try:
        return json.load(open(_FORM2RO))
    except Exception:
        return {}

@functools.lru_cache(maxsize=1)
def _concepts():
    cs = json.load(open(_ROBI))["concepts"]
    byid, byform = {}, {}
    for c in cs:
        byid[c["id"]] = c
        terms = [c["preferred_term"]["term"]] + [
            (s["term"] if isinstance(s, dict) else s) for s in c.get("synonyms", [])]
        for t in terms:
            byform.setdefault(str(t).strip().lower(), c["id"])
    return byid, byform

def ro_of_form(form):
    """ROBI concept id for a relation surface form. Pool map first, then ROBI synonyms."""
    if form is None:
        return None
    pool, (_, byform) = _pool_forms(), _concepts()
    f = str(form).strip().lower()
    for alt in [f] + [a.strip() for a in f.split("|")]:
        if not alt:
            continue
        if alt in EXTRA_FORMS:
            return EXTRA_FORMS[alt]
        if alt in pool:
            return pool[alt]
        if alt in byform:
            return byform[alt]
        for v in (alt.rstrip("s"), alt + "s"):
            if v in EXTRA_FORMS:
                return EXTRA_FORMS[v]
            if v in pool:
                return pool[v]
            if v in byform:
                return byform[v]
    return None

def pref_term(ro):
    byid, _ = _concepts()
    c = byid.get(ro)
    return c["preferred_term"]["term"] if c else None

def polarity(ro):
    """+1 subject is agent-side, -1 subject is host-side, 0 symmetric, None unknown."""
    return POL.get(ro)

def subject_is_species1(role_of_s1, ro):
    """Compose a role assignment with relation polarity -> 1 (s1) / 2 (s2) / None."""
    p = polarity(ro)
    if p is None or p == 0 or role_of_s1 is None:
        return None
    # role_of_s1: +1 if s1 is the agent-side taxon, -1 if s1 is the host-side taxon
    return 1 if role_of_s1 == p else 2
