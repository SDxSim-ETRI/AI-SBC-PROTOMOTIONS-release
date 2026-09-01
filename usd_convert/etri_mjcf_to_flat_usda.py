# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""MJCF → **flat USDA** 변환기 (ProtoMotions IsaacLab 백엔드용).

왜 또 하나의 변환기인가
-----------------------
`convert_robot_mjcf_to_usda.py` 는 Isaac Sim 의 MJCF 임포터를 감싸 **계층형**
자산(`<name>.usda` 래퍼 + `configuration/{base,physics,robot,sensor}.usd` payload +
variantSet)을 만든다. 그 구조로는 ProtoMotions 의 IsaacLab 백엔드에서 접촉 센서가
붙지 않는다(2026-08-06 실측):

    RuntimeError: Sensor at path '.../worldBody/L_Ankle' could not find any bodies
                  with contact reporter API.

`activate_contact_sensors()` 가 payload 안의 body 에 `PhysxContactReportAPI` 를 심지
못하기 때문이다. `--make-instanceable` 을 끄고 articulation root 를 정리해도 남는다.

**정상 동작하는 자산은 계층형이 아니다.** upstream 이 배포한
`protomotions/data/assets/usd/smpl_humanoid.usda` 는 단일 flat `.usda` 로,
24개 body 에 `PhysicsRigidBodyAPI` 가 텍스트로 직접 박혀 있다(커밋 57f98a963).
이 스크립트는 **그 형식을 그대로 생성**한다.

생성 구조 (참조 자산과 동일)
----------------------------
```
def Xform "<name>" ( delete apiSchemas = [ArticulationRootAPI, PhysxArticulationAPI] )
{
    def Xform "bodies"
    {
        def Xform "Pelvis" ( prepend apiSchemas = [RigidBody, PhysxRigidBody, Mass,
                                                   FilteredPairs, ArticulationRoot,
                                                   PhysxArticulation, AnimationData] )
        { physics:mass / physics:density / xformOp:transform
          def "collisions" { def Cube|Capsule|Sphere "_geom_N" (PhysicsCollisionAPI…) }
          def "visuals"    { def Cube|Capsule|Sphere "_geom_N" (MaterialBindingAPI) } }
        ...
    }
    def "joints"
    {
        def PhysicsJoint "L_Hip" ( PhysxJointAPI, PhysicsLimitAPI:*, PhysicsDriveAPI:* )
        { drive/limit/armature, mjcf:rot{X,Y,Z}:name, physics:body0/1, localPos0/1 }
        ...
    }
}
```

★ 질량은 **명시**한다 — 참조 자산과 다른 점
-------------------------------------------
참조 자산은 body 에 `physics:density = 0` 만 두고 `physics:mass` 가 없다. 그러면 PhysX 가
콜라이더 부피 × 기본 밀도로 질량을 만든다. 맨몸 SMPL 은 MJCF geom 이 `density="1000"`
이라 우연히 맞지만, **외골격 슈트는 MJCF 에 `mass="0.1977"` 처럼 실측 질량이 직접
지정**돼 있어 그 방식으로는 재현되지 않는다(부피×1000 으로 계산하면 슈트가 몇 배 무거워
진다 — eval XML 에서 실제로 겪은 버그다).

그래서 각 body 에 `float physics:mass = <MuJoCo body_mass>` 를 쓴다. PhysX 는 authored
mass 를 density 보다 우선하므로 MuJoCo 총질량과 **정확히** 일치한다.
검증: exosuitHS 68.014 kg (맨몸 63.314 + 슈트 4.700).

★ 시각 전용 geom(`contype=0`)은 콜라이더로 쓰지 않는다
-----------------------------------------------------
슈트 파트는 질량만 기여하고 충돌에는 참여하지 않는다(`exosuit_spec` 의
`suit_collision=False`). 그래서 `collisions` 에는 `contype != 0` 인 geom 만 넣고,
`visuals` 에는 전부 넣는다. 질량은 위 `physics:mass` 가 이미 반영한다.

사용법
  cd /home/user/ProtoMotions
  /home/user/venv_newton/bin/python3 usd_convert/etri_mjcf_to_flat_usda.py \\
      tasks_for_smpl/mimic_smpl/data/assets/mjcf_newton_exosuitHS/smpl_humanoid_exosuitHS_for_train.xml \\
      -o tasks_for_smpl/mimic_smpl/data/assets/usd_isaaclab_exosuitHS/smpl_humanoid_exosuitHS_for_train.usda

  Isaac Sim 이 필요 없다 — mujoco 만 쓰므로 venv_newton 에서 돈다.

옵션
  --name NAME        루트 prim 이름 (기본: 출력 파일 stem)
  --stiffness / --damping / --armature   MJCF 값이 없을 때의 관절 기본값
  --check            생성 후 MuJoCo 총질량과 대조만 하고 끝낸다
"""

import argparse
from pathlib import Path

import mujoco
import numpy as np

# MuJoCo geom type → (USD prim 타입, 콜라이더 근사)
GEOM_SPHERE, GEOM_CAPSULE, GEOM_CYLINDER, GEOM_BOX = 2, 3, 5, 6
APPROX = {GEOM_BOX: "boundingCube", GEOM_SPHERE: "boundingSphere",
          GEOM_CAPSULE: "convexHull", GEOM_CYLINDER: "convexHull"}

ROOT_APIS = ('["PhysicsRigidBodyAPI", "PhysxRigidBodyAPI", "PhysicsMassAPI", '
             '"PhysicsFilteredPairsAPI", "PhysicsArticulationRootAPI", '
             '"PhysxArticulationAPI", "AnimationDataAPI"]')
LINK_APIS = ('["PhysicsRigidBodyAPI", "PhysxRigidBodyAPI", "PhysicsMassAPI", '
             '"PhysicsFilteredPairsAPI"]')
AXES = ("X", "Y", "Z")


def quat_to_mat(q_wxyz):
    """MuJoCo wxyz → 3x3."""
    w, x, y, z = q_wxyz
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def mat4(R, t, scale=None):
    """USD `matrix4d` 문자열.

    ★ USD/GfMatrix4d 는 **행벡터 규약**(v' = v·M)이다. MuJoCo 의 R 은 열벡터 규약
      (v' = R·v)이므로 **전치해서** 저장해야 한다. 전치를 빼면 회전이 거울처럼 뒤집혀
      대칭 도형(캡슐·박스)에서는 티가 안 나지만 **사선 캡슐의 기울기가 반대로** 나온다.
      실측(2026-08-06): 허리↔대퇴골구 연결 캡슐 축이
          MuJoCo  [ 0.0119, -0.0906, 0.9958]
          R 저장  [-0.0119,  0.0906, 0.9958]   ← x·y 부호 반전
          R.T 저장[ 0.0119, -0.0906, 0.9958]   ✔
    """
    M = np.eye(4)
    RS = R if scale is None else R @ np.diag(scale)
    M[:3, :3] = RS.T
    M[3, :3] = t
    rows = ", ".join("(" + ", ".join(repr(float(v)) for v in M[i]) + ")" for i in range(4))
    return f"( {rows} )"


def geom_shape(m, g):
    """(USD prim 타입, 속성 dict, extent) — MuJoCo geom size 규약을 USD 로 옮긴다."""
    t = int(m.geom_type[g])
    s = m.geom_size[g]
    if t == GEOM_BOX:
        # MJCF size = 반길이. USD Cube 는 size 2 정규 큐브 + scale 로 표현한다.
        e = s[:3]
        return "Cube", {"extent": e}, e
    if t == GEOM_SPHERE:
        r = float(s[0])
        return "Sphere", {"radius": r}, np.array([r, r, r])
    if t in (GEOM_CAPSULE, GEOM_CYLINDER):
        r, half = float(s[0]), float(s[1])
        prim = "Capsule" if t == GEOM_CAPSULE else "Cylinder"
        # MuJoCo 캡슐/실린더의 축은 **로컬 z**. USD 는 axis 토큰으로 지정한다.
        ext = np.array([r, r, half + (r if t == GEOM_CAPSULE else 0.0)])
        return prim, {"radius": r, "height": 2 * half, "axis": "Z"}, ext
    raise SystemExit(f"지원하지 않는 geom type {t} (geom {g}). "
                     f"mesh 는 flat usda 로 낼 수 없다 — 학습 XML 은 primitive 만 써야 한다.")


def emit_geom(out, m, g, idx, indent, collision, gname=None, rgba=None):
    prim, attrs, ext = geom_shape(m, g)
    pad = " " * indent
    apis = ('["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]' if collision
            else '["MaterialBindingAPI"]')
    R = quat_to_mat(m.geom_quat[g])
    t = m.geom_pos[g]
    # Cube 는 scale 로 반길이를 표현한다(참조 자산과 동일).
    scale = attrs["extent"] if prim == "Cube" else None
    # ★ prim 이름에 MJCF geom 이름을 넣는다.
    #   IsaacLab 의 `apply_humanoid_visual_materials()` 는 **prim 경로 문자열**로 색을
    #   고른다(`exo_main`, `hip_ring` … 포함 여부). `_geom_N` 으로만 쓰면 전부 skin 색으로
    #   떨어져 슈트와 인체가 구분되지 않는다(2026-08-06 실측: 58 prim 전부 아이보리).
    #   무명 geom(인체 primitive)은 참조 자산 규약대로 `_geom_N` 을 유지한다.
    pname = f"_geom_{idx}" if not gname else f"_geom_{idx}_{gname}"
    out.append(f'{pad}def {prim} "{pname}" (')
    out.append(f'{pad}    apiSchemas = {apis}')
    out.append(f'{pad})')
    out.append(f'{pad}{{')
    if "axis" in attrs:
        out.append(f'{pad}    uniform token axis = "{attrs["axis"]}"')
    out.append(f'{pad}    float3[] extent = [({-ext[0]}, {-ext[1]}, {-ext[2]}), '
               f'({ext[0]}, {ext[1]}, {ext[2]})]')
    if "height" in attrs:
        out.append(f'{pad}    double height = {attrs["height"]}')
    if collision:
        out.append(f'{pad}    uniform token physics:approximation = "{APPROX[int(m.geom_type[g])]}"')
        out.append(f'{pad}    uniform token purpose = "guide"')
    if "radius" in attrs:
        out.append(f'{pad}    double radius = {attrs["radius"]}')
    if not collision and rgba is not None:
        # ★ MJCF rgba → USD displayColor. 이게 없으면 인체와 슈트가 **같은 색**으로 나와
        #   시각 검증에서 파트를 구분할 수 없다(2026-08-06 실측). 알파도 함께 넘긴다.
        r, g_, b = (float(v) for v in rgba[:3])
        out.append(f'{pad}    color3f[] primvars:displayColor = [({r}, {g_}, {b})]')
        if float(rgba[3]) < 1.0:
            out.append(f'{pad}    float[] primvars:displayOpacity = [{float(rgba[3])}]')
    if gname:
        # ★ 원래 MJCF geom 이름을 남긴다. prim 이름은 참조 자산 규약(`_geom_N`)을 지켜야
        #   IsaacLab 의 재질 적용이 동작하므로, 이름은 custom token 으로 보존한다.
        out.append(f'{pad}    custom token mjcf:name = "{gname}"')
    out.append(f'{pad}    matrix4d xformOp:transform = {mat4(R, t, scale)}')
    out.append(f'{pad}    uniform token[] xformOpOrder = ["xformOp:transform"]')
    out.append(f'{pad}}}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mjcf")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--stiffness", type=float, default=800.0)
    ap.add_argument("--damping", type=float, default=80.0)
    ap.add_argument("--armature", type=float, default=0.02)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(a.mjcf)
    out_path = Path(a.out)
    name = a.name or out_path.stem
    nm = lambda t, i: mujoco.mj_id2name(m, t, i)

    bodies = [i for i in range(1, m.nbody)]                 # 0 = world
    root = bodies[0]
    print(f"[1/3] MJCF 로드 {Path(a.mjcf).name}")
    print(f"      body {len(bodies)}  geom {m.ngeom}  joint {m.njnt}  "
          f"총질량 {m.body_mass.sum():.4f} kg")

    L = ["#usda 1.0", "(", f'    defaultPrim = "{name}"', '    upAxis = "Z"', ")", "",
         f'def Xform "{name}" (',
         '    delete apiSchemas = ["PhysicsArticulationRootAPI", "PhysxArticulationAPI"]',
         ")", "{",
         '    def Xform "bodies"', "    {",
         "        quatd xformOp:orient = (1, 0, 0, 0)",
         "        double3 xformOp:scale = (1, 1, 1)",
         "        double3 xformOp:translate = (0, 0, 0)",
         '        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", '
         '"xformOp:scale"]', ""]

    print("[2/3] bodies 작성")
    for b in bodies:
        bname = nm(mujoco.mjtObj.mjOBJ_BODY, b)
        apis = ROOT_APIS if b == root else LINK_APIS
        kw = "prepend apiSchemas" if b == root else "apiSchemas"
        L += [f'        def Xform "{bname}" (', f'            {kw} = {apis}',
              "        )", "        {"]
        # ★ 질량 명시 — 슈트의 실측 mass 를 그대로 옮긴다 (docstring 참고)
        L += [f"            float physics:mass = {float(m.body_mass[b])}",
              "            float physics:density = 0",
              f"            matrix4d xformOp:transform = "
              f"{mat4(quat_to_mat(m.body_quat[b]), m.body_pos[b])}",
              '            uniform token[] xformOpOrder = ["xformOp:transform"]', ""]
        gids = list(range(m.body_geomadr[b], m.body_geomadr[b] + m.body_geomnum[b]))
        col = [g for g in gids if m.geom_contype[g] != 0]
        L += ['            def "collisions"', "            {"]
        for i, g in enumerate(col):
            emit_geom(L, m, g, i, 16, True,
                      nm(mujoco.mjtObj.mjOBJ_GEOM, g))
        L += ["            }", "", '            def "visuals"', "            {"]
        for i, g in enumerate(gids):
            emit_geom(L, m, g, i, 16, False,
                      nm(mujoco.mjtObj.mjOBJ_GEOM, g), m.geom_rgba[g])
        L += ["            }", "        }", ""]
    L += ["    }", ""]

    print("[3/3] joints 작성")
    L += ['    def "joints"', "    {"]
    njoint = 0
    for b in bodies:
        jids = [j for j in range(m.njnt) if m.jnt_bodyid[j] == b
                and int(m.jnt_type[j]) == mujoco.mjtJoint.mjJNT_HINGE]
        if not jids:
            continue                                        # free joint(루트) 는 관절이 아니다
        bname = nm(mujoco.mjtObj.mjOBJ_BODY, b)
        pname = nm(mujoco.mjtObj.mjOBJ_BODY, m.body_parentid[b])
        # MuJoCo 힌지 축(1 0 0 / 0 1 0 / 0 0 1)을 USD rot{X,Y,Z} 로 대응시킨다.
        axis_map = {}
        for j in jids:
            ax = np.abs(m.jnt_axis[j]).argmax()
            axis_map[AXES[ax]] = j
        schemas = ['"PhysxJointAPI"', '"PhysicsLimitAPI:transX"', '"PhysicsLimitAPI:transY"',
                   '"PhysicsLimitAPI:transZ"']
        for A in AXES:
            if A in axis_map:
                schemas += [f'"PhysicsLimitAPI:rot{A}"', f'"PhysxLimitAPI:rot{A}"',
                            f'"PhysicsDriveAPI:rot{A}"']
        L += [f'        def PhysicsJoint "{bname}" (',
              "            apiSchemas = [" + ", ".join(schemas) + "]",
              "        )", "        {"]
        for A in AXES:
            if A not in axis_map:
                continue
            j = axis_map[A]
            kp = float(m.jnt_stiffness[j]) or a.stiffness
            kd = a.damping
            if m.dof_damping[m.jnt_dofadr[j]] > 0:
                kd = float(m.dof_damping[m.jnt_dofadr[j]])
            L += [f"            float drive:rot{A}:physics:damping = {kd}",
                  f"            float drive:rot{A}:physics:maxForce = 3.4028235e38",
                  f"            float drive:rot{A}:physics:stiffness = {kp}",
                  f'            uniform token drive:rot{A}:physics:type = "force"']
        for A in AXES:
            if A not in axis_map:
                continue
            j = axis_map[A]
            lo, hi = (np.degrees(m.jnt_range[j]) if m.jnt_limited[j] else (-180.0, 180.0))
            L += [f"            float limit:rot{A}:physics:high = {float(hi)}",
                  f"            float limit:rot{A}:physics:low = {float(lo)}"]
        # 병진은 잠근다 — high < low 가 USD 의 "잠금" 관용 표현이다(참조 자산과 동일).
        for A in AXES:
            L += [f"            float limit:trans{A}:physics:high = -1",
                  f"            float limit:trans{A}:physics:low = 1"]
        for A in AXES:
            if A in axis_map:
                jn = nm(mujoco.mjtObj.mjOBJ_JOINT, axis_map[A])
                L += [f'            custom token mjcf:rot{A}:name = "{jn}"']
        L += [f"            rel physics:body0 = </{name}/bodies/{pname}>",
              f"            rel physics:body1 = </{name}/bodies/{bname}>",
              "            float physics:breakForce = 3.4028235e38",
              "            float physics:breakTorque = 3.4028235e38",
              "            point3f physics:localPos0 = ("
              + ", ".join(f"{float(v)}" for v in m.body_pos[b]) + ")",
              "            point3f physics:localPos1 = (0, 0, 0)",
              "            quatf physics:localRot0 = (1, 0, 0, 0)",
              "            quatf physics:localRot1 = (1, 0, 0, 0)",
              f"            float physxJoint:armature = "
              f"{float(m.dof_armature[m.jnt_dofadr[jids[0]]]) or a.armature}"]
        for A in AXES:
            if A in axis_map:
                j = axis_map[A]
                kp = float(m.jnt_stiffness[j]) or a.stiffness
                kd = float(m.dof_damping[m.jnt_dofadr[j]]) or a.damping
                L += [f"            float physxLimit:rot{A}:damping = {kd}",
                      f"            float physxLimit:rot{A}:stiffness = {kp}"]
        L += ["        }", ""]
        njoint += 1
    L += ["    }", "}", ""]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(L))
    print(f"      body {len(bodies)}  joint {njoint}")
    print(f"완료! {out_path}  ({out_path.stat().st_size/1024:.0f} KB)")

    # ── 검증: 우리가 써 넣은 질량 합이 MuJoCo 와 같은가 ──────────────
    txt = out_path.read_text()
    written = [float(x.split("=")[1]) for x in txt.splitlines()
               if "float physics:mass =" in x]
    print(f"검증  physics:mass 항목 {len(written)}개  합 {sum(written):.4f} kg  "
          f"(MuJoCo {m.body_mass.sum():.4f})  "
          f"{'일치 ✔' if abs(sum(written)-m.body_mass.sum()) < 1e-6 else '★ 불일치'}")
    print(f"      RigidBodyAPI {txt.count('PhysicsRigidBodyAPI')}  "
          f"콜라이더 {txt.count('PhysicsCollisionAPI')}  "
          f"PhysicsJoint {txt.count('def PhysicsJoint')}")
    print(f"      displayColor {txt.count('primvars:displayColor')}  "
          f"geom 이름 보존 {txt.count('custom token mjcf:name')}")


if __name__ == "__main__":
    main()
