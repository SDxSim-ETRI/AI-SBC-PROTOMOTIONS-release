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
"""
외골격 슈트 **명세(데이터)**. 슈트가 늘어나도 스크립트는 하나로 유지한다.

두 슈트 모두 같은 SMPL 24 body / 69 DOF 골격에 붙으므로 모션·골격·스킨·렌더러·
robot config 골격은 전부 공유된다. 슈트마다 다른 것은 이 파일의 6가지뿐이다:

  mesh_dir      : CAD STL 폴더 (mesh/exosuit_hs, mesh/exosuit_cr)
  cad_anchor    : CAD→MJCF 앵커 + 회전(quat) — CAD 좌표계를 SMPL 골격에 맞추는 값
  parts         : primitive geom 목록 (name, body, attrs, mass[, eval_override])
                  eval_override 를 주면 eval XML 에서만 그 attrs 로 대체된다
                  (예: 발목링 train=구 / eval=원통)
  cad_meshes    : 시각용 CAD mesh 와 붙는 body
  exo_actuators : 보조 모터가 물리는 관절 + 피크 토크
  eval_hidden   : eval 렌더에서 감출 primitive (CAD mesh 가 대신 보여주는 파트)
  suit_collision : 슈트 geom 을 충돌에 참여시킬지 (기본 False = **질량만**)
                  MuJoCo·Newton 모두 contype=0 이어도 질량·관성은 그대로 계산된다
                  (검증 2026-07-29: 두 엔진 71.314 kg 동일). 평지 보행 학습에서는
                  슈트가 지면·환경과 부딪힐 일이 없고, 몸과의 겹침은 착용 구조상
                  당연하므로 충돌을 끄는 것이 레퍼런스 모션과 싸우지 않게 한다.
  contact_excludes : <contact><exclude> 로 뺄 body 쌍 (suit_collision=True 일 때만 의미)
                  ※ **질량과 무관하다** — 질량은 geom `mass=` 에서 나오고, exclude 는
                    그 두 body 쌍의 접촉 검사만 끈다. 다른 body/지면과는 계속 충돌한다.

질량은 density 가 아니라 **kg 직접 지정**이다. 속 찬 도형 × density 로 계산하면
실물과 크게 어긋난다(HS 의 경우 20.6kg vs 실제 1.8kg).

────────────────────────────────────────────────────────────────────────
hs = 힙 전용 슈트 (2026-08-03 개정).  **CR 형상에서 무릎 이하를 뺀 것**, 실측 4.700 kg.
     모터는 힙 2개(Unitree GO-M8010-6, 23.7 Nm).  task_dir=mimic_smpl_exosuitHS.
     ※ 구 HyperShell CAD 정의(1.80 kg, CAD mesh 3종)는 사용자 지시로 폐기했다.
cr = Cosmo Robotics 전신 외골격.  실측 6.300 kg, 모터 힙+무릎 4개.
────────────────────────────────────────────────────────────────────────
"""


# ── 손/손목 간섭 회피 (요청) ─────────────────────────────────────────
# 걷기(CMU 07_04)에서 팔 스윙이 힙 모터 구·ㄷ자 팔·허벅지 box 를 관통한다.
# 몸쪽 이동량을 스윕한 결과: 0mm→25.6 / 10mm→16.0 / 20mm→7.3 / 25mm→3.0 / **30mm→0**.
# → 좌우 측방 파트를 **43mm** 안으로 넣는다(두께 교환 반영 후 재조정).
# 대가: 힙 모터가 CAD 실측 위치(±191mm)보다 43mm 안쪽이 되어 대퇴 표면 안으로 들어간다.
#       HS 의 CAD mesh 는 힙축 앵커 기준이라 여전히 ±191mm 이므로 primitive 와 43mm 차이가 난다
#       (eval 은 CAD 를 보여주므로 시각적으로는 문제 없음).
_INSET = 0.043   # 30mm → 43mm (두께 교환으로 box 가 13mm 더 나와 손 간섭 재발)

# ── HS 질량 (합 4.700 kg) — CR 실측에서 **무릎 이하를 뺀** 값 ──────────
# 2026-08-03 사용자 지시로 기존 HyperShell CAD 정의(1.80 kg, task_dir=mimic_smpl)를
# **폐기하고** CR 형상 파생 힙 전용 슈트로 교체했다. 파트별 값은 CR 과 동일하고
# 무릎 이하 8개(knee_motor·shank_box·calf_ring·ankle_ring ×2, 1.600 kg)만 없다.
#   가방 1.370 + 허리밴드(bar+arm×2) 1.690 + 허벅지파트(모터+링·박스)×2 1.640
#   = **4.700 kg**  (= CR 6.300 의 "종아리 위쪽" 실측값과 일치)
_HS_MASS = dict(bag=1.370, bar=0.9389, arm=0.3756, hip_motor=0.530,
                thigh_box=0.1977, ring=0.0923)   # 합 4.700

# 허벅지 링(스트랩) 규칙 — CR 과 동일:
#   지름 = 대퇴 +1cm (r 0.0615/0.0606 → 0.0665/0.0656), 반높이 0.02(2/3),
#   대퇴 실린더 아랫끝(반원 직전) z=-0.3002/-0.3061, **대퇴축 중심** 정렬.
#   축 중심에 두면 좌우 링이 pelvis 기준 ±95mm 로 벌어져 서로 부딪히지 않는다
#   (예전 r=0.082 / pos z=-0.2 는 좌우 링이 22mm 관통했다).

# ── Cosmo Robotics 질량 (요청 조정: 모터 각 0.6 / ㄷ바 1.0 / 가방 1.0 /
#                          허벅지box 0.6 / 종아리box 0.3) ─────────────────────
# 참고 모델: ~/x2_human_mujoco/x2.xml (하지 외골격, 힙+무릎 구동 = CR 과 같은 급)
#   backpack 8.063 / upper_thigh 2.184(편측) / upper_shank 1.611(편측) / 실물 합 15.653 kg
#   ※ drive_inertia 0.100×4 는 액추에이터 반사관성 더미 — 실물 질량 아님(우리는 관절
#     armature=0.02 로 표현하므로 질량에 넣지 않는다).
# 세그먼트 합 비교:
#   가방+ㄷ프레임 = 1.00 + 1.00 + 0.40×2 = 2.80   (X2 backpack 8.063 — 1/3 이하)
#   허벅지 조립   = 0.60 + 0.60 + 0.28    = 1.48   (X2 upper_thigh 2.184 — 68%)
#   종아리 조립   = 0.60 + 0.30 + 0.11 + 0.11 = 1.12  (X2 upper_shank 1.611 — 70%)
#   합계 **8.00 kg**  → X2 15.653 의 51%. 경량 보행보조급.
# ⚠️ 실측 BOM 받으면 이 표만 교체.
# ★ CR 질량 = **실측 확정** (2026-07-31, 코스모로보틱스 제공). 총 6.300 kg.
#
#   유닛 단위 실측:
#     허벅지파트 (모터 530 + 링·박스 290) × 2 = 1.640
#     종아리파트 (모터 530 + 링2·박스 270) × 2 = 1.600
#     허리밴드파트 (ㄷ 프레임 + 스트랩)          = 1.690
#     가방 (배터리 404 + 임베디드 + wifi 모듈)   = 1.370
#   검산: 종아리 위쪽(가방+허리밴드+허벅지) = 1.370+1.690+1.640 = **4.700**
#         + 종아리 1.600                          = **6.300** ✓
#
#   모터 530 g 은 Unitree GO-M8010-6 스펙과 일치(최대 토크 23.7 Nm).
#   유닛 **내부** 세부 배분(bar:arm, box:ring)은 기존 모델 비율을 유지했다 —
#   실측이 유닛 단위로만 주어졌기 때문이다.
#
#   이전 임의값(총 8.000) 대비: 다리 한쪽 2.600 → 1.620 kg (−38%), 상체 +0.370.
#   질량이 **위로 이동** → 측방 오프셋 다리 질량이 만드는 외전/회전축 부하가 줄어
#   슈트 무게 대가(ablation 실측 +4.55%)가 감소할 것으로 예상.
_CR_MASS = dict(bag=1.370, bar=0.9389, arm=0.3756, hip_motor=0.530, thigh_box=0.1977,
                knee_motor=0.530, shank_box=0.1558, ring=0.0923,
                ring_calf=0.0571, ring_ankle=0.0571)   # 합 6.300

SUITS = {
    "hs": dict(
        label="exosuitHS",
        # ★ 2026-08-05: 자산을 `mimic_smpl` 로 옮겼다. S1(슈트 착용 인체제어기)이
        #   거기서 학습하므로 슈트 XML 이 필요하고, 사본을 두면 두 벌이 갈라진다.
        #   `mimic_smpl_exosuitHS` 는 슈트제어기(S2) 코드 전용이 됐다.
        task_dir="mimic_smpl",
        vendor="HyperShell",
        note="힙 전용(무릎 이하 없음). CR 형상에서 파생. 실측 4.700 kg.",
        mesh_dir="exosuit_hs",
        # ── CAD(STL) 사용 ────────────────────────────────────────────────
        # train 은 primitive 그대로, **eval 만** HyperShell CAD 로 보여준다.
        #   hip_motor.stl  ㄷ자 허리 프레임 + 힙 모터 구  (Pelvis)
        #   left_leg.stl   / right_leg.stl  허벅지 스트럿  (L_Hip / R_Hip)
        # CAD 가 없는 파트(힙 링, 가방)는 eval 에서도 primitive 로 남는다.
        #
        # 좌표 정합: primitive 배치가 원래 HyperShell XML(2026-07-29)과 **동일**하므로
        # CAD 가 그대로 맞는다. 힙 body pos(L -0.0068 0.0695 -0.0914 / R -0.0043
        # -0.0677 -0.0905)와 mesh pos 가 일치하는 것을 확인했다.
        # cad_anchor_mm 은 보관된 원본 eval XML 의 mesh pos 에서 역산했다
        # (_backup/hypershell_1.8kg_20260729/ — 왕복 오차 5.6e-17 m).
        # cad_rot 은 cad_quat 의 회전행렬과 같다(벡터용/MJCF geom 속성용으로 이원화).
        cad_anchor_mm=(-86.0399, -230.85, 227.5897),
        pelvis_anchor=(-0.0055, 0.0, -0.091),          # 좌우 힙 회전축 중점
        cad_quat="0.6905832 0.1660015 0.1640944 0.6845519",
        cad_rot=((0.008923378844, -0.891000312597, 0.453914987925),
                 (0.999959986387, 0.007664326974, -0.004613427874),
                 (0.000631612792, 0.453937992511, 0.891033052148)),
        suit_collision=False,   # 슈트 geom = 질량만 (충돌 제외)
        parts=[
            # [ETRI 2026-08-26] 가방을 뒤로 55 mm 이동 (-0.08 → -0.135).
            #   이유: 앞면이 몸 캡슐을 파고들어 렌더에서 살에 묻혀 보였다.
            #   실측 관통(rest pose, 가방 앞면 → 몸 캡슐 최단거리):
            #     Chest -52.8 mm / Torso -35.5 mm / Spine -22.5 mm
            #   물리 영향: 슈트 geom 은 contype=0(질량만)이라 충돌은 원래 없었다.
            #   질량 이동 영향도 미미 — 1.370 kg × 55 mm / 68.014 kg = CoM 1.1 mm.
            # [ETRI 2026-08-26] 가방: 두께 절반 + 등쪽으로 기울임.
            #   ① 반두께 0.04 → 0.02 (두께 80 → 40 mm). 질량은 1.370 kg 유지.
            #   ② 기울기 y축 -12° — 상단이 뒤로, 하단이 앞으로 → 허리가 등에 붙는다.
            #   ③ pos x -0.08 → -0.090 (10 mm 만 이동)
            #   실측 여유(rest pose, 앞면 → 몸 캡슐): 하단 +4.7 / 중간 +10.7 / 상단 +4.9 mm
            #   (수정 전: Chest -52.8 / Torso -35.5 / Spine -22.5 mm 관통)
            #   물리 영향 없음 — 슈트 geom 은 contype=0(질량만).
            ("exo_main_col", "Torso",
             dict(type="box", size="0.02 0.1 0.1374", pos="-0.090 0 0.1213",
                  quat="0.994521895 0.0 -0.104528463 0.0"),
             _HS_MASS["bag"]),
            ("waist_hip_bar", "Pelvis",
             dict(type="capsule", fromto="-0.1166 -0.074 0.0436 -0.1166 0.074 0.0436", size="0.040"),
             _HS_MASS["bar"]),
            ("waist_hip_arm_l", "Pelvis",
             dict(type="capsule", fromto="-0.0055 0.148 -0.091 -0.1166 0.074 0.0436", size="0.030"),
             _HS_MASS["arm"]),
            ("waist_hip_arm_r", "Pelvis",
             dict(type="capsule", fromto="-0.0055 -0.148 -0.091 -0.1166 -0.074 0.0436", size="0.030"),
             _HS_MASS["arm"]),
            ("left_hip_motor", "L_Hip",
             dict(type="sphere", size="0.0385", pos="0.0013 0.0785 0.0004"),
             _HS_MASS["hip_motor"]),
            ("right_hip_motor", "R_Hip",
             dict(type="sphere", size="0.0385", pos="-0.0012 -0.0803 -0.0005"),
             _HS_MASS["hip_motor"]),
            ("left_thigh_box", "L_Hip",
             dict(type="box", size="0.030 0.02 0.1883", pos="-0.0004 0.0717 -0.1878", quat="0.99983 -0.01806 0.00438 0"),
             _HS_MASS["thigh_box"]),
            ("right_thigh_box", "R_Hip",
             dict(type="box", size="0.030 0.02 0.1909", pos="-0.0026 -0.0740 -0.1913", quat="0.99986 0.01638 0.00367 0"),
             _HS_MASS["thigh_box"]),
            ("left_hip_ring", "L_Hip",
             dict(type="capsule", fromto="-0.0031 0.0238 -0.2603 -0.0036 0.0274 -0.3001", size="0.069"),
             _HS_MASS["ring"]),
            ("right_hip_ring", "R_Hip",
             dict(type="capsule", fromto="-0.0062 -0.0266 -0.2662 -0.0071 -0.0306 -0.3060", size="0.0681"),
             _HS_MASS["ring"]),
        ],
        # (mesh 이름, STL 파일, 붙는 body, 굴곡전용 렌더)
        # 굴곡전용 = 다리 CAD 를 힙 y(굴곡)축만 따라 런타임에 그린다. 힙 x/z 까지
        # 따르면 모터 반구 2개가 어긋나 보인다(exosuit_leg_mesh.LegMeshes).
        cad_meshes=[("exo_hip_frame", "hip_motor.stl", "Pelvis", False),
                    ("exo_thigh_l", "left_leg.stl", "L_Hip", True),
                    ("exo_thigh_r", "right_leg.stl", "R_Hip", True)],
        cad_meshes_pending=False,
        # 보조 모터 = **힙 2개만**. 피크는 CR 과 같은 Unitree GO-M8010-6 (23.7 Nm).
        exo_actuators=[("exo_hip_l", "L_Hip_y", 23.7),
                       ("exo_hip_r", "R_Hip_y", 23.7)],
        # eval 에서 감추는 primitive = **가방(exo_main_col)을 뺀 전부**.
        # hip_motor.stl(ㄷ자 허리 프레임)과 left/right_leg.stl(대퇴골 모터·허벅지 옆
        # box·허벅지 링)이 이 파트들을 모두 담고 있으므로, primitive 를 같이 보여주면
        # CAD 와 이중으로 겹쳐 보인다. 가방만 CAD 가 없어 primitive 로 남긴다.
        # ★ left/right_hip_ring(허벅지 원통)은 이 목록에서 **뺐다** — 보이게 둔다.
        #   CAD STL 에 대응 파트가 없고(STL 의 밴드는 무릎밴드다), 허벅지 스트랩이
        #   감기는 위치를 보여주는 유일한 표시다(2026-08-06 요청). 색은 CAD 와 같은
        #   회색으로, 높이는 STL 밴드 실측(89.9 / 89.5 mm)에 맞췄다.
        eval_hidden=["waist_hip_bar", "waist_hip_arm_l", "waist_hip_arm_r",
                     "left_hip_motor", "right_hip_motor",
                     "left_thigh_box", "right_thigh_box"],
        # 손/손목이 힙 측방 파트를 스치는 것은 1-DOF 하드웨어를 3-DOF 관절에 붙인
        # 한계이고 레퍼런스가 슈트 없이 캡처된 것이라 불가피하다.
        contact_excludes=[("L_Hand", "L_Hip"), ("L_Wrist", "L_Hip"),
                          ("R_Hand", "R_Hip"), ("R_Wrist", "R_Hip"),
                          ("L_Hand", "Pelvis"), ("L_Wrist", "Pelvis"),
                          ("R_Hand", "Pelvis"), ("R_Wrist", "Pelvis")],
    ),
    # ── Cosmo Robotics 전신 외골격 ─────────────────────────────────
    # CAD(STL) 미수령 → **primitive 도형만** 으로 구성하고, eval 에서도 그 도형을
    # 그대로 보여준다(cad_meshes/eval_hidden 이 비어 있으므로 자동으로 그렇게 된다).
    # HS 에서 삭제한 전신 파트(가방·무릎모터·종아리box·종아리링·발목링)를 여기서 되살린다.
    #
    # 링/가방 치수는 **몸 캡슐에서 역산**했다 (mjcf/smpl_humanoid_for_train.xml):
    #   가방     : 허리(Torso) 캡슐 하단 z=-0.0161 까지 내림 → 반높이 0.18 → center 0.1639
    #   허벅지링 : 지름 = 대퇴 **+1.5cm** (r 0.0615/0.0606 → 0.069/0.0681)
    #              (+1cm 은 살 속에 잠겨 보였다)
    #              길이 = 이전 몸통의 2/3 (반높이 0.03 → 0.02), 대퇴 실린더 아랫끝(반원 직전)
    #              **대퇴축 중심**에 둔다. box 에 붙이려고 옆으로 당기지 않는다(요청) —
    #              링은 사지를 감싸는 것이 우선이고, box 와의 접촉은 우연에 맡긴다.
    #   ※ 종아리링·발목링도 같은 원칙: 축 중심 정렬만, 옆 box 에 억지로 붙이지 않는다.
    #
    # ★ 링/원통은 **pos+size 가 아니라 `fromto` 로 정의**한다. pos 방식은 축이 body 로컬 Z
    #   에 고정되는데, 대퇴/정강이 축은 그보다 5.2~6.6° 기울어 있어 링이 사지에 대해
    #   삐뚤어져 보인다. fromto 로 두 점을 사지 축 위에 두면 축이 정확히 일치하고,
    #   길이도 |p2-p1| 로 직접 정해진다(capsule 은 여기에 반원 2개가 더 붙는다).
    #   eval 에서 type 만 cylinder 로 바꾸면 길이가 그대로 유지되어 "반원 뺀 몸통" 이 된다.
    #   종아리링 : 지름 = 정강이 +1cm (0.0541 → 0.0591), 반높이 0.035 → **0.0233 (2/3)**,
    #              정강이축 중심 정렬 → 종아리옆 box 에 물린다
    #   발목링   : **구(sphere)**. 원통이면 걷는 동안 모양이 이상해 보인다(요청).
    #              정강이 캡슐 끝 반원과 **동심**(pos = 캡슐 끝), 지름 = 정강이 +1cm
    #              → 회전 대칭이라 자세가 바뀌어도 형상이 변하지 않는다.
    #
    # ※ 질량은 **전부 임의 placeholder** (Cosmo BOM 미수령). 합 9.08 kg.
    #    실측 받으면 _CR_MASS 만 교체.
    "cr": dict(
        label="exosuitCR",
        task_dir="mimic_smpl_exosuitCR",  # 학습 태스크 폴더로 이동(자립)
        vendor="Cosmo Robotics",
        note="전신 외골격. CAD 미수령 → primitive 도형만 (eval 도 도형 사용). 질량 placeholder.",
        mesh_dir="exosuit_cr",
        cad_anchor_mm=None,
        pelvis_anchor=(-0.0055, 0.0, -0.091),
        cad_quat=None,
        cad_rot=None,
        suit_collision=False,   # 슈트 geom = 질량만 (충돌 제외)
        parts=[
            # 가방 (배터리 + 임베디드 + wifi)
            # ※ 렌더 착시 주의 (2026-08-03 확인): 슈트 rgba 알파가 0.55(반투명)라
            #   **정면 시점에서 가방이 몸통을 투과해 가슴에 파란 판으로 보인다.**
            #   실제 형상은 정상이다 — 깊이 8 × 폭 20 × 높이 27.5 cm 수직 판이고
            #   Torso 원점보다 x −8 cm(등 쪽), 기립자세 회전은 단위행렬.
            #   후면 프레임(walk_cmu_38_04)에서 등에 정상 부착된 것을 확인했다.
            #   HF 클립(squat·stepinplace 등)은 카메라가 계속 정면이라 이 착시가
            #   두드러진다. 사용자 판단으로 **투명도는 그대로 유지**한다.
            #   높이: 아래는 허리(Torso) 캡슐 하단 z=-0.0161, 위는 **어깨(Thorax) 캡슐
            #   하단 z=+0.2638 보다 5mm 아래**(+0.2588) → 반높이 0.1374 (기존 0.18).
            #   기존 높이(360mm)는 걷기 전 프레임에서 어깨를 54.7mm 관통했다.
            ("exo_main_col", "Torso",
             dict(type="box", size="0.04 0.1 0.1374", pos="-0.08 0 0.1213"), _CR_MASS["bag"]),
            # 사선 ㄷ 프레임 (HS 와 동일 형상)
            ("waist_hip_bar", "Pelvis",
             dict(type="capsule", fromto="-0.1166 -0.074 0.0436 -0.1166 0.074 0.0436", size="0.040"),
             _CR_MASS["bar"]),
            ("waist_hip_arm_l", "Pelvis",
             dict(type="capsule", fromto="-0.0055 0.148 -0.091 -0.1166 0.074 0.0436", size="0.030"),
             _CR_MASS["arm"]),
            ("waist_hip_arm_r", "Pelvis",
             dict(type="capsule", fromto="-0.0055 -0.148 -0.091 -0.1166 -0.074 0.0436", size="0.030"),
             _CR_MASS["arm"]),
            # 힙 (대퇴골) 모터 + 허벅지
            #   ※ 정면 두께(y): 허벅지box 0.02 / 종아리box 0.007 — 요청에 따라 서로 교환
            #     (이전: 허벅지 0.007 / 종아리 0.02)
            ("left_hip_motor", "L_Hip",
             dict(type="sphere", size="0.0385", pos="0.0013 0.0785 0.0004"), _CR_MASS["hip_motor"]),
            ("right_hip_motor", "R_Hip",
             dict(type="sphere", size="0.0385", pos="-0.0012 -0.0803 -0.0005"), _CR_MASS["hip_motor"]),
            ("left_thigh_box", "L_Hip",
             dict(type="box", size="0.030 0.02 0.1883", pos="-0.0004 0.0717 -0.1878",
                  quat="0.99983 -0.01806 0.00438 0"), _CR_MASS["thigh_box"]),
            ("right_thigh_box", "R_Hip",
             dict(type="box", size="0.030 0.02 0.1909", pos="-0.0026 -0.0740 -0.1913",
                  quat="0.99986 0.01638 0.00367 0"), _CR_MASS["thigh_box"]),
            ("left_hip_ring", "L_Hip",
             dict(type="capsule", fromto="-0.0031 0.0238 -0.2603 -0.0036 0.0274 -0.3001", size="0.069"), _CR_MASS["ring"]),
            ("right_hip_ring", "R_Hip",
             dict(type="capsule", fromto="-0.0062 -0.0266 -0.2662 -0.0071 -0.0306 -0.3060", size="0.0681"), _CR_MASS["ring"]),
            # 무릎 모터 + 종아리
            #   ※ 종아리옆 box 는 아래끝이 **발목 원통까지** 내려온다 (z -0.34 → -0.36)
            # 무릎 모터 구: 힙 모터(0.0385)와 **같은 크기로 통일** (이전 0.05 로 아래가 더 컸음)
            ("left_knee_motor", "L_Knee",
             dict(type="sphere", size="0.0385", pos="0 0.06 0"), _CR_MASS["knee_motor"]),
            ("right_knee_motor", "R_Knee",
             dict(type="sphere", size="0.0385", pos="0 -0.06 0"), _CR_MASS["knee_motor"]),
            ("left_shank_box", "L_Knee",
             dict(type="box", size="0.025 0.007 0.17", pos="-0.02 0.055 -0.19"), _CR_MASS["shank_box"]),
            ("right_shank_box", "R_Knee",
             dict(type="box", size="0.025 0.007 0.17", pos="-0.02 -0.0525 -0.19"), _CR_MASS["shank_box"]),
            ("left_calf_ring", "L_Knee",
             dict(type="capsule", fromto="-0.0139 -0.0043 -0.1269 -0.0190 -0.0059 -0.1731", size="0.0591"), _CR_MASS["ring_calf"]),
            ("right_calf_ring", "R_Knee",
             dict(type="capsule", fromto="-0.0135 0.0051 -0.1268 -0.0184 0.0069 -0.1732", size="0.0591"), _CR_MASS["ring_calf"]),
            # 발목 링 — 정강이 캡슐 아래쪽 반원을 덮는 **정강이 스트랩**이다.
            # ★ 2026-07-29 수정: 부모를 L_Ankle(발) → **L_Knee(정강이)**.
            #   발에 붙어 있으면 발목이 꺾일 때 링이 발과 함께 회전한다("걸을 때
            #   모양이 이상함"의 원인 — 구로 바꿔 증상만 가려져 있었다). 질량
            #   0.11kg 도 정강이가 아니라 발에 실려 스윙 다이내믹스가 틀렸다.
            #   좌표는 발 프레임 → 무릎 프레임 변환(L_Ankle pos 만큼 더함)으로
            #   **중립자세 형상이 1:1 동일**하다. 변환 결과가 정강이 캡슐의 아래쪽
            #   끝점과 정확히 일치한다(L: -0.0350 -0.0109 -0.3184).
            # train=구(자세 무관), eval=원통(길이 54.1 → 34.1mm, 요청대로 2cm 짧게)
            ("left_ankle_ring", "L_Knee",
             dict(type="sphere", pos="-0.0350 -0.0109 -0.3184", size="0.0591"), _CR_MASS["ring_ankle"],
             dict(type="cylinder", fromto="-0.0350 -0.0109 -0.3184 -0.0387 -0.0121 -0.3523", size="0.0591")),
            ("right_ankle_ring", "R_Knee",
             dict(type="sphere", pos="-0.0338 0.0126 -0.3187", size="0.0591"), _CR_MASS["ring_ankle"],
             dict(type="cylinder", fromto="-0.0338 0.0126 -0.3187 -0.0374 0.0139 -0.3526", size="0.0591")),
        ],
        cad_meshes=[],          # CAD 없음 → eval 도 primitive 그대로
        cad_meshes_pending=True,
        # 피크 = **실제 스펙** Unitree GO-M8010-6 최대 토크 23.7 Nm (2026-07-31 확정).
        # 출처: ~/Downloads/Unitree_GO-M8010-6_Specification.md
        # ※ 이 값은 MJCF <actuator> 용(문서·MuJoCo 검증). 학습에서 실제로 쓰는 피크는
        #   mimic/mlp_newton_actionnet.py 의 EXO_MOTORS 다 — 두 곳을 함께 맞춰야 한다.
        # ※ 스펙에 **연속 정격이 없다**. 23.7 은 최대값이고 보행은 지속 동작이라
        #   발열 한계 확인 필요(측정 RMS 11.1 Nm = 피크의 47%).
        exo_actuators=[("exo_hip_l", "L_Hip_y", 23.7), ("exo_hip_r", "R_Hip_y", 23.7),
                       ("exo_knee_l", "L_Knee_y", 23.7), ("exo_knee_r", "R_Knee_y", 23.7)],
        eval_hidden=[],         # 가릴 CAD 가 없으므로 전부 표시
        # 접촉 제외 (질량 영향 없음) — 학습구현.md 자기충돌 스윕 기준
        contact_excludes=[("L_Hand", "L_Hip"), ("L_Wrist", "L_Hip"),
                          ("R_Hand", "R_Hip"), ("R_Wrist", "R_Hip"),
                          ("L_Hand", "Pelvis"), ("L_Wrist", "Pelvis"),
                          ("R_Hand", "Pelvis"), ("R_Wrist", "Pelvis"),
                          # 좌우 발목링(r=0.0591)이 모을 때 31.3mm 관통.
                          # exclude 는 body 단위라 발 box 끼리의 접촉도 함께 빠지지만,
                          # 레퍼런스에서 이미 0.4mm 겹치는 marginal 접촉이라 무해하다.
                          ("L_Ankle", "R_Ankle"),
                          ("L_Shoulder", "Torso"), ("R_Shoulder", "Torso")],
    ),
}


def get(suit: str) -> dict:
    if suit not in SUITS:
        raise SystemExit(f"알 수 없는 슈트 '{suit}'. 가능: {list(SUITS)}")
    s = SUITS[suit]
    if not s["parts"]:
        raise SystemExit(f"'{suit}' ({s['vendor']}) 명세가 비어 있습니다 — {__file__} 를 먼저 채우세요.\n"
                         f"  {s['note']}")
    return s


def total_mass(suit: str) -> float:
    return sum(pt[3] for pt in SUITS[suit]["parts"])   # pt = (name, body, attrs, mass[, eval_override])


# ── 경로 헬퍼 ────────────────────────────────────────────────────────
from pathlib import Path as _Path   # noqa: E402

_TASK_ROOT = _Path(__file__).parent.parent


def paths(suit: str) -> dict:
    """슈트별 자산 경로.  <task_dir>/data/assets/{mjcf_newton_<label>, mesh/…}

    슈트마다 자산 루트가 다르다(학습 태스크 폴더로 자립시키기 위함):
      hs → tasks_for_smpl/mimic_smpl/data/assets            (공용, 아직 미분리)
      cr → tasks_for_smpl/mimic_smpl_exosuitCR/data/assets  (학습 태스크 폴더)
    """
    s = SUITS[suit]
    label = s["label"]
    _ASSETS = _TASK_ROOT / s["task_dir"] / "data/assets"
    d = _ASSETS / f"mjcf_newton_{label}"
    return dict(
        assets=_ASSETS, dir=d, label=label,
        train=d / f"smpl_humanoid_{label}_for_train.xml",
        eval_skeleton=d / f"smpl_humanoid_{label}_for_eval_skeleton.xml",
        eval_lbskin=d / f"smpl_humanoid_{label}_for_eval_lbskin.xml",
        usd=_ASSETS / f"usd_isaaclab_{label}",
        # eval XML 의 meshdir 는 ../mesh/skeleton 이므로 CAD 는 그 기준 상대경로로 참조
        mesh_rel=f"../{s['mesh_dir']}",
        mesh_abs=_ASSETS / "mesh" / s["mesh_dir"],
    )


def cad_rot_np(suit: str):
    """CAD→MJCF 회전행렬 (numpy). CAD 없는 슈트면 None."""
    import numpy as np
    r = SUITS[suit]["cad_rot"]
    return None if r is None else np.array(r, float)


def has_cad(suit: str) -> bool:
    return bool(SUITS[suit]["cad_meshes"])


def flex_only_meshes(suit: str):
    """힙 굴곡만 따라야 하는 CAD mesh: [(geom name, body, 굴곡 관절)]"""
    out = []
    for name, _fn, body, flex in SUITS[suit]["cad_meshes"]:
        if flex:
            out.append((f"mesh_{name}", body, f"{body}_y"))
    return out
