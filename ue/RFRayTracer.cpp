// RFRayTracer.cpp · 见 RFRayTracer.h 头部说明
#include "RFRayTracer.h"
#include "Engine/World.h"
#include "Components/StaticMeshComponent.h"
#include "Materials/MaterialInterface.h"
#include "PhysicalMaterials/PhysicalMaterial.h"
#include "DrawDebugHelpers.h"
#include "Misc/FileHelper.h"
#include "CollisionQueryParams.h"

ARFRayTracer::ARFRayTracer()
{
	PrimaryActorTick.bCanEverTick = false;
	RootComponent = CreateDefaultSubobject<USceneComponent>(TEXT("Root"));
}

static float Log10f_(float x) { return FMath::Loge(FMath::Max(x, 1e-12f)) / FMath::Loge(10.0f); }

float ARFRayTracer::FreeSpaceLossDb(float DistCm) const
{
	const float d_km = FMath::Max(DistCm / 100.0f / 1000.0f, 1e-6f);   // cm→m→km
	return 20.0f * Log10f_(d_km) + 20.0f * Log10f_(FrequencyMHz) + 32.44f;
}

bool ARFRayTracer::Trace(const FVector& A, const FVector& B, FHitResult& Hit) const
{
	FCollisionQueryParams P(FName(TEXT("RFTrace")), bTraceComplex);
	P.bReturnPhysicalMaterial = true;
	P.bReturnFaceIndex = true;                 // 需 Project Settings: Support UV From Hit Results
	P.AddIgnoredActor(this);

	// 自动跳过 UE Landscape 地形（模板自带、非我们的电磁材质；删不掉也没关系）
	const FVector Dir = (B - A).GetSafeNormal();
	FVector Start = A;
	for (int32 k = 0; k < 32; ++k)
	{
		FHitResult H;
		if (!GetWorld()->LineTraceSingleByChannel(H, Start, B, ECC_Visibility, P))
			return false;
		const UPrimitiveComponent* C = H.GetComponent();
		if (C && C->GetClass()->GetName().Contains(TEXT("Landscape")))
		{
			Start = H.Location + Dir * 1.0f;                       // 跳过地形，从命中点后继续
			if (FVector::DotProduct(B - Start, Dir) <= 0.f) return false;  // 已越过终点
			continue;
		}
		Hit = H;
		return true;
	}
	return false;
}

// 命中材质名 → 电磁参数：DataTable 优先，否则内置回退表（与 ue_materials.csv 一致）
bool ARFRayTracer::KeyToEM(const FString& RawName, float& Pen, float& Eps, float& Sig) const
{
	const FString N = RawName.ToLower();
	auto Has = [&N](const TCHAR* s) { return N.Contains(s); };

	// 先查 DataTable（行名如 EM_concrete）
	if (MaterialTable)
	{
		static const TCHAR* Rows[] = { TEXT("EM_background"), TEXT("EM_vegetation"), TEXT("EM_ground_road"),
									   TEXT("EM_concrete"), TEXT("EM_roof"), TEXT("EM_glass"), TEXT("EM_metal") };
		for (const TCHAR* R : Rows)
		{
			if (N.Contains(FString(R).ToLower().Mid(3)))   // 匹配 concrete/glass/...
			{
				if (FRFMaterialRow* Row = MaterialTable->FindRow<FRFMaterialRow>(FName(R), TEXT("")))
				{
					Pen = Row->penetration_loss_db; Eps = Row->eps_r; Sig = Row->sigma; return true;
				}
			}
		}
	}

	// 内置回退表（ITU-R P.2040 @3.5GHz）
	if (Has(TEXT("metal")))       { Pen = 40.f; Eps = 1.0f;  Sig = 1e7f;   return true; }
	if (Has(TEXT("glass")))       { Pen = 4.f;  Eps = 6.27f; Sig = 0.0043f;return true; }
	if (Has(TEXT("concrete")))    { Pen = 12.f; Eps = 5.31f; Sig = 0.0326f;return true; }
	if (Has(TEXT("roof")))        { Pen = 8.f;  Eps = 4.0f;  Sig = 0.01f;  return true; }
	if (Has(TEXT("ground"))||Has(TEXT("road"))||Has(TEXT("asphalt")))
	                              { Pen = 30.f; Eps = 8.0f;  Sig = 0.02f;  return true; }
	if (Has(TEXT("veg"))||Has(TEXT("wood"))||Has(TEXT("tree")))
	                              { Pen = 6.f;  Eps = 1.5f;  Sig = 0.05f;  return true; }
	if (Has(TEXT("background")))  { Pen = 0.f;  Eps = 1.0f;  Sig = 0.0f;   return true; }
	return false;
}

// 命中面片的材质名(原始)：先物理材质，再命中面片的材质槽，最后组件名
FString ARFRayTracer::HitMaterialName(const FHitResult& Hit) const
{
	FString Key;
	if (UPhysicalMaterial* PM = Hit.PhysMaterial.Get())
		Key = PM->GetName();
	if (Key.IsEmpty() || !Key.Contains(TEXT("EM_")))
	{
		if (UStaticMeshComponent* SMC = Cast<UStaticMeshComponent>(Hit.GetComponent()))
		{
			if (Hit.FaceIndex >= 0)
			{
				int32 Sec = 0;
				if (UMaterialInterface* MI = SMC->GetMaterialFromCollisionFaceIndex(Hit.FaceIndex, Sec))
					Key = MI->GetName();
			}
		}
	}
	if (Key.IsEmpty())
		if (UPrimitiveComponent* C = Hit.GetComponent()) Key = C->GetName();
	return Key;
}

// 归一到 7 类之一
FString ARFRayTracer::CleanClass(const FString& Raw) const
{
	const FString N = Raw.ToLower();
	static const TCHAR* Keys[] = { TEXT("background"), TEXT("vegetation"), TEXT("ground_road"),
								   TEXT("concrete"), TEXT("roof"), TEXT("glass"), TEXT("metal") };
	for (const TCHAR* K : Keys) if (N.Contains(K)) return FString(K);
	if (N.Contains(TEXT("road")) || N.Contains(TEXT("ground"))) return TEXT("ground_road");
	if (N.Contains(TEXT("veg")) || N.Contains(TEXT("wood")))    return TEXT("vegetation");
	return Raw;   // 认不出就原样返回
}

void ARFRayTracer::LookupEM(const FHitResult& Hit, float& Pen, float& Eps, float& Sig) const
{
	Pen = 0.f; Eps = 1.f; Sig = 0.f;
	KeyToEM(HitMaterialName(Hit), Pen, Eps, Sig);
}

// TX→RX 逐层穿透累加（镜像 simple_pathloss.py 的第二遍）
float ARFRayTracer::PenetrationAlong(const FVector& TX, const FVector& RX) const
{
	const FVector Dir = (RX - TX).GetSafeNormal();
	const float Eps = 1.0f;                      // 1cm 步进，避免自命中
	FVector O = TX; float Total = 0.f;
	for (int32 k = 0; k < MaxPenetrations; ++k)
	{
		float Remain = FVector::Dist(O, RX);
		FHitResult Hit;
		if (!Trace(O + Dir * Eps, RX, Hit)) break;
		if (Hit.Distance >= Remain - Eps)       break;   // 命中点已在 RX 之后 → 停
		float p, er, sg; LookupEM(Hit, p, er, sg);
		Total += p;
		O = Hit.Location + Dir * Eps;
	}
	return Total;
}

// ============ 点对点探针：几条 TX→目标 射线，识别命中材质（简单测试）============
void ARFRayTracer::RunProbe()
{
	UWorld* W = GetWorld();
	if (!W) return;
	ClearDebug();

	const FVector TX = GetActorLocation();
	DrawDebugSphere(W, TX, 200.f, 12, FColor::Magenta, true, -1.f, 0, 4.f);   // TX 品红球

	int32 n = 0;
	for (int32 i = 0; i < ProbeTargets.Num(); ++i)
	{
		AActor* T = ProbeTargets[i];
		if (!T) continue;
		++n;
		const FVector RX = T->GetActorLocation();
		DrawDebugSphere(W, RX, 150.f, 10, FColor::Cyan, true, -1.f, 0, 3.f);   // 目标青球

		FHitResult H;
		if (!Trace(TX, RX, H))    // 直达无遮挡
		{
			DrawDebugLine(W, TX, RX, FColor::Green, true, -1.f, 0, 3.f);
			DrawDebugString(W, (TX + RX) * 0.5f, TEXT("LoS clear"), nullptr, FColor::Green, -1.f, false, ProbeTextScale);
			UE_LOG(LogTemp, Warning, TEXT("[RF][Probe %d] LoS 直达，无遮挡"), i);
			continue;
		}

		// 首个命中材质
		const FString Cls = CleanClass(HitMaterialName(H));
		float p, er, sg; LookupEM(H, p, er, sg);
		DrawDebugLine(W, TX, H.Location, FColor::Yellow, true, -1.f, 0, 3.f);            // 首击段=黄
		DrawDebugLine(W, H.Location, RX, FColor(120, 120, 120), true, -1.f, 0, 1.f);     // 之后=灰(示意)
		DrawDebugString(W, H.Location,
			FString::Printf(TEXT("%s (eps=%.2f, pen=%.0fdB)"), *Cls, er, p),
			nullptr, FColor::White, -1.f, false, ProbeTextScale);
		UE_LOG(LogTemp, Warning,
			TEXT("[RF][Probe %d] 首击=%s  d=%.1fm  eps_r=%.2f sigma=%.4g 穿透=%.1fdB"),
			i, *Cls, H.Distance / 100.f, er, sg, p);

		// 列出 TX→RX 穿过的每层材质
		if (bProbeListLayers)
		{
			const FVector Dir = (RX - TX).GetSafeNormal();
			const float Step = 1.0f; FVector O = TX; float total = 0.f; int32 layer = 0;
			FString seq;
			for (int32 k = 0; k < MaxPenetrations; ++k)
			{
				const float Remain = FVector::Dist(O, RX);
				FHitResult Hk;
				if (!Trace(O + Dir * Step, RX, Hk)) break;
				if (Hk.Distance >= Remain - Step) break;
				const FString c = CleanClass(HitMaterialName(Hk));
				float pk, ek, sk; LookupEM(Hk, pk, ek, sk);
				total += pk; ++layer;
				seq += (layer > 1 ? TEXT(" -> ") : TEXT("")) + c;
				O = Hk.Location + Dir * Step;
			}
			UE_LOG(LogTemp, Warning, TEXT("[RF][Probe %d] 穿过 %d 层: %s  累计穿透=%.1f dB"),
				i, layer, seq.IsEmpty() ? TEXT("(仅首击/直达)") : *seq, total);
		}
	}

	if (n == 0)
	{
		UE_LOG(LogTemp, Warning, TEXT("[RF][Probe] ProbeTargets 为空：请在细节面板往 ProbeTargets 拖几个目标Actor"));
	}
	else
	{
		UE_LOG(LogTemp, Warning, TEXT("[RF][Probe] 完成 %d 条射线（黄=首击段, 青球=目标, 品红球=TX）。材质见上方日志"), n);
	}
}

void ARFRayTracer::RunCoverage()
{
	UWorld* W = GetWorld();
	if (!W) return;
	ClearDebug();

	const FVector TX = GetActorLocation();
	TArray<FString> Lines;
	Lines.Add(TEXT("x_cm,y_cm,z_cm,los,pathloss_db,prx_dbm"));

	int32 nValid = 0, nLoS = 0;
	float MinPL = TNumericLimits<float>::Max(), MaxPL = -TNumericLimits<float>::Max();
	struct FRxRec { FVector P; bool LoS; float PL, Prx; };
	TArray<FRxRec> Recs;

	for (int32 iy = 0; iy < GridN; ++iy)
	for (int32 ix = 0; ix < GridN; ++ix)
	{
		const float fx = (GridN > 1) ? (float)ix / (GridN - 1) : 0.5f;
		const float fy = (GridN > 1) ? (float)iy / (GridN - 1) : 0.5f;
		const float x = GridCenter.X + FMath::Lerp(-GridExtentX, GridExtentX, fx);
		const float y = GridCenter.Y + FMath::Lerp(-GridExtentY, GridExtentY, fy);

		// 从高处往下找地表
		FHitResult G;
		const FVector Top(x, y, TX.Z + SkyTraceTopCm);
		const FVector Bot(x, y, TX.Z - 200000.0f);
		if (!Trace(Top, Bot, G)) continue;                 // 无地表 → 跳过
		const FVector RX = G.Location + FVector(0, 0, RxHeightCm);
		++nValid;

		const float Dist = FVector::Dist(TX, RX);
		const float FSPL = FreeSpaceLossDb(Dist);

		// LoS 判定
		FHitResult H; bool bBlocked = false;
		if (Trace(TX, RX, H))
			bBlocked = (H.Distance < Dist - 2.0f);

		float PL = FSPL;
		if (bBlocked) PL += PenetrationAlong(TX, RX); else ++nLoS;
		const float Prx = TxPowerDbm + TxGainDbi + RxGainDbi - PL;

		MinPL = FMath::Min(MinPL, PL); MaxPL = FMath::Max(MaxPL, PL);
		Recs.Add({ RX, !bBlocked, PL, Prx });
		Lines.Add(FString::Printf(TEXT("%.1f,%.1f,%.1f,%d,%.2f,%.2f"),
			RX.X, RX.Y, RX.Z, bBlocked ? 0 : 1, PL, Prx));
	}

	// 上色绘制（按路损 jet）
	if (bDrawDebug)
	{
		const float Span = FMath::Max(MaxPL - MinPL, 1e-3f);
		for (const FRxRec& R : Recs)
		{
			const float t = (R.PL - MinPL) / Span;
			DrawDebugPoint(W, R.P, DebugPointSize, JetColor(t), true, -1.f, 0);
		}
		DrawDebugSphere(W, TX, 300.f, 12, FColor::Magenta, true, -1.f, 0, 8.f);
	}

	if (!OutputCsvPath.IsEmpty())
		FFileHelper::SaveStringToFile(FString::Join(Lines, TEXT("\n")), *OutputCsvPath);

	UE_LOG(LogTemp, Warning, TEXT("[RF] Coverage: valid=%d LoS=%d(%.1f%%) PL=[%.1f,%.1f]dB -> %s"),
		nValid, nLoS, nValid ? 100.f * nLoS / nValid : 0.f, MinPL, MaxPL, *OutputCsvPath);
}

void ARFRayTracer::RunRayFan()
{
	UWorld* W = GetWorld();
	if (!W) return;
	const FVector TX = GetActorLocation();
	const float Len = 100000.0f;                 // 1km 射线

	for (int32 ie = 0; ie < FanElevation; ++ie)
	for (int32 ia = 0; ia < FanAzimuth; ++ia)
	{
		const float az = 2.f * UE_PI * ia / FMath::Max(FanAzimuth, 1);
		const float el = -UE_HALF_PI * (0.15f + 0.8f * ie / FMath::Max(FanElevation - 1, 1)); // 下半球
		const FVector Dir(FMath::Cos(el) * FMath::Cos(az), FMath::Cos(el) * FMath::Sin(az), FMath::Sin(el));
		const FVector End = TX + Dir * Len;

		FHitResult H;
		if (Trace(TX, End, H))
		{
			float p, er, sg; LookupEM(H, p, er, sg);
			const bool bGroundish = (p <= 6.f);      // 地面/植被视作可达
			const FColor C = bGroundish ? FColor::Green : FColor::Red;
			DrawDebugLine(W, TX, H.Location, C, true, -1.f, 0, DebugRayThick);

			if (bReflectBounce && !bGroundish)       // 命中建筑 → 画一次镜面反射
			{
				const FVector Refl = Dir - 2.0f * FVector::DotProduct(Dir, H.Normal) * H.Normal; // 反射向量(手算)
				FHitResult H2;
				const FVector RE = H.Location + Refl * Len;
				if (Trace(H.Location + Refl * 2.f, RE, H2))
					DrawDebugLine(W, H.Location, H2.Location, FColor::Orange, true, -1.f, 0, DebugRayThick);
				else
					DrawDebugLine(W, H.Location, RE, FColor::Orange, true, -1.f, 0, DebugRayThick * 0.5f);
			}
		}
	}
	DrawDebugSphere(W, TX, 300.f, 12, FColor::Magenta, true, -1.f, 0, 8.f);
	UE_LOG(LogTemp, Warning, TEXT("[RF] RayFan drawn from TX %s"), *TX.ToString());
}

void ARFRayTracer::ClearDebug()
{
	if (UWorld* W = GetWorld())
		FlushPersistentDebugLines(W);
}

FColor ARFRayTracer::JetColor(float t)
{
	t = FMath::Clamp(t, 0.f, 1.f);
	const float r = FMath::Clamp(1.5f - FMath::Abs(4.f * t - 3.f), 0.f, 1.f);
	const float g = FMath::Clamp(1.5f - FMath::Abs(4.f * t - 2.f), 0.f, 1.f);
	const float b = FMath::Clamp(1.5f - FMath::Abs(4.f * t - 1.f), 0.f, 1.f);
	return FColor(uint8(r * 255), uint8(g * 255), uint8(b * 255));
}
