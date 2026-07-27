// RFRayTracer.h · UE5 射频几何射线追踪器（面向低空通信仿真）
// 放置于关卡后：Actor 位置即 TX；对导入的材质网格做 LineTrace，
// 命中读物理材质/材质槽 → 查 ITU-R P.2040 电磁参数 → LoS/NLoS + 穿透损耗 → 覆盖热图 + CSV。
// 与 Python simple_pathloss.py 同一物理模型(FSPL + 穿透)，供三引擎交叉验证。
//
// 用法：拷到 <你的UE工程>/Source/<模块>/ 下，重新生成工程并编译；
//       在关卡放一个 RFRayTracer，Details 面板点 “Run Coverage” / “Run Ray Fan”。
#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "Engine/DataTable.h"
#include "RFRayTracer.generated.h"

/** DataTable 行：与 ue_materials.csv 对应（RowName 用材质名，如 EM_concrete）。可选，不填有内置回退表。 */
USTRUCT(BlueprintType)
struct FRFMaterialRow : public FTableRowBase
{
	GENERATED_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "RF") float eps_r = 1.0f;                // 相对介电常数
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "RF") float sigma = 0.0f;               // 电导率 S/m
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "RF") float penetration_loss_db = 0.0f; // 穿透损耗 dB
};

UCLASS()
class ARFRayTracer : public AActor
{
	GENERATED_BODY()

public:
	ARFRayTracer();

	// ---- 无线电参数（与 configs/radio_params.yaml 一致）----
	UPROPERTY(EditAnywhere, Category = "RF") float FrequencyMHz = 3500.0f;
	UPROPERTY(EditAnywhere, Category = "RF") float TxPowerDbm  = 23.0f;
	UPROPERTY(EditAnywhere, Category = "RF") float TxGainDbi   = 0.0f;
	UPROPERTY(EditAnywhere, Category = "RF") float RxGainDbi   = 0.0f;

	// ---- RX 接收点网格（UE 厘米；水平面，落到地表上方 RxHeightCm）----
	UPROPERTY(EditAnywhere, Category = "RF|Grid") int32   GridN        = 60;       // NxN
	UPROPERTY(EditAnywhere, Category = "RF|Grid") float   GridExtentX  = 34000.0f; // 半宽 X(cm)，见 ue_anchor.json bbox
	UPROPERTY(EditAnywhere, Category = "RF|Grid") float   GridExtentY  = 26000.0f; // 半宽 Y(cm)
	UPROPERTY(EditAnywhere, Category = "RF|Grid") FVector2D GridCenter = FVector2D(0.f, 0.f);
	UPROPERTY(EditAnywhere, Category = "RF|Grid") float   RxHeightCm   = 150.0f;   // 地表上方(≈1.5m)
	UPROPERTY(EditAnywhere, Category = "RF|Grid") float   SkyTraceTopCm= 30000.0f; // 从 TX 之上多高往下找地表

	// ---- 传播选项 ----
	UPROPERTY(EditAnywhere, Category = "RF") int32 MaxPenetrations = 30;      // NLoS 穿透累加最多层数
	UPROPERTY(EditAnywhere, Category = "RF") bool  bTraceComplex   = true;    // 复杂碰撞(逐三角)，必须为真才能逐面片

	// ---- 射线扇(可视化)----
	UPROPERTY(EditAnywhere, Category = "RF|Fan") int32 FanAzimuth   = 72;     // 方位角采样
	UPROPERTY(EditAnywhere, Category = "RF|Fan") int32 FanElevation = 12;     // 俯仰采样(下半球)
	UPROPERTY(EditAnywhere, Category = "RF|Fan") bool  bReflectBounce = true; // 命中建筑后画一次镜面反射

	// ---- 点对点探针（简单测试：几条 TX→目标 的射线，识别命中材质）----
	// 用法：在几栋楼上各放一个 Actor（空Actor/球都行），拖进这个数组；点 Run Probe。
	UPROPERTY(EditAnywhere, Category = "RF|Probe") TArray<AActor*> ProbeTargets;
	UPROPERTY(EditAnywhere, Category = "RF|Probe") bool  bProbeListLayers = true; // 日志列出穿过的每层材质
	UPROPERTY(EditAnywhere, Category = "RF|Probe") float ProbeTextScale   = 1.5f; // 命中材质名文字大小

	// ---- 调试绘制 ----
	UPROPERTY(EditAnywhere, Category = "RF|Debug") bool  bDrawDebug     = true;
	UPROPERTY(EditAnywhere, Category = "RF|Debug") float DebugRayThick  = 1.5f;
	UPROPERTY(EditAnywhere, Category = "RF|Debug") float DebugPointSize = 14.0f;

	// ---- 材质表 + 输出 ----
	UPROPERTY(EditAnywhere, Category = "RF|Material") UDataTable* MaterialTable = nullptr;
	UPROPERTY(EditAnywhere, Category = "RF|Output")   FString OutputCsvPath =
		TEXT("E:/lowalt3d/outputs/scene02_pgsr/ue/coverage_ue.csv");

	// ---- 编辑器按钮（放好 Actor 直接点，无需 Play）----
	UFUNCTION(CallInEditor, Category = "RF") void RunProbe();      // 点对点射线 + 材质识别（简单测试，推荐先用这个）
	UFUNCTION(CallInEditor, Category = "RF") void RunCoverage();
	UFUNCTION(CallInEditor, Category = "RF") void RunRayFan();
	UFUNCTION(CallInEditor, Category = "RF") void ClearDebug();

private:
	// 命中 → 电磁参数（先物理材质名，再退回材质槽名，最后内置表）
	void  LookupEM(const FHitResult& Hit, float& OutPenDb, float& OutEps, float& OutSigma) const;
	bool  KeyToEM(const FString& RawName, float& OutPenDb, float& OutEps, float& OutSigma) const;
	FString HitMaterialName(const FHitResult& Hit) const;         // 命中面片的材质名(原始)
	FString CleanClass(const FString& Raw) const;                 // 归一到 7 类之一(background/concrete/…)
	float FreeSpaceLossDb(float DistCm) const;               // FSPL(dB)
	float PenetrationAlong(const FVector& TX, const FVector& RX) const; // NLoS 穿透累加(dB)
	bool  Trace(const FVector& A, const FVector& B, FHitResult& Hit) const;
	static FColor JetColor(float t);                          // 0..1 → 蓝-绿-红
};
