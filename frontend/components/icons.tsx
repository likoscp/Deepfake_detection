import React from "react";

interface IconProps extends React.SVGProps<SVGSVGElement> {
  size?: number;
}

const svgBase = (size: number, strokeWidth = "1.6") =>
  ({
    viewBox: "0 0 24 24",
    width: size,
    height: size,
    fill: "none",
    stroke: "currentColor",
    strokeWidth,
    strokeLinecap: "round" as const,
    strokeLinejoin: "round" as const,
  } as const);

export function ShieldIcon({ size = 16, ...p }: IconProps) {
  return (
    <svg {...svgBase(size)} {...p}>
      <path d="M12 3l8 3v5c0 5-3.5 8.5-8 10-4.5-1.5-8-5-8-10V6l8-3z" />
    </svg>
  );
}

export function CameraIcon({ size = 16, ...p }: IconProps) {
  return (
    <svg {...svgBase(size)} {...p}>
      <path d="M3 7h4l2-2h6l2 2h4v12H3z" />
      <circle cx="12" cy="13" r="3.5" />
    </svg>
  );
}

export function MailIcon({ size = 16, ...p }: IconProps) {
  return (
    <svg {...svgBase(size)} {...p}>
      <rect x="3" y="5" width="18" height="14" rx="2" />
      <path d="M3 7l9 6 9-6" />
    </svg>
  );
}

export function CheckIcon({ size = 16, ...p }: IconProps) {
  return (
    <svg {...svgBase(size, "2")} {...p}>
      <path d="M4 12l5 5L20 6" />
    </svg>
  );
}

export function XIcon({ size = 16, ...p }: IconProps) {
  return (
    <svg {...svgBase(size, "2")} {...p}>
      <path d="M6 6l12 12M6 18L18 6" />
    </svg>
  );
}

export function WarnIcon({ size = 16, ...p }: IconProps) {
  return (
    <svg {...svgBase(size)} {...p}>
      <path d="M12 3l10 18H2L12 3z" />
      <path d="M12 10v5" />
      <circle cx="12" cy="18" r="0.6" fill="currentColor" />
    </svg>
  );
}

export function ArrowRightIcon({ size = 16, ...p }: IconProps) {
  return (
    <svg {...svgBase(size, "1.8")} {...p}>
      <path d="M5 12h14M13 6l6 6-6 6" />
    </svg>
  );
}

export function CpuIcon({ size = 16, ...p }: IconProps) {
  return (
    <svg {...svgBase(size, "1.5")} {...p}>
      <rect x="6" y="6" width="12" height="12" rx="1.5" />
      <rect x="9" y="9" width="6" height="6" />
      <path d="M9 3v3M15 3v3M9 18v3M15 18v3M3 9h3M3 15h3M18 9h3M18 15h3" />
    </svg>
  );
}

export function LayersIcon({ size = 16, ...p }: IconProps) {
  return (
    <svg {...svgBase(size, "1.5")} {...p}>
      <path d="M12 3l9 5-9 5-9-5 9-5z" />
      <path d="M3 13l9 5 9-5" />
      <path d="M3 17l9 5 9-5" />
    </svg>
  );
}

export function RefreshIcon({ size = 16, ...p }: IconProps) {
  return (
    <svg {...svgBase(size)} {...p}>
      <path d="M3 12a9 9 0 0115.5-6.3L21 8" />
      <path d="M21 3v5h-5" />
      <path d="M21 12a9 9 0 01-15.5 6.3L3 16" />
      <path d="M3 21v-5h5" />
    </svg>
  );
}

export function SparkIcon({ size = 16, ...p }: IconProps) {
  return (
    <svg {...svgBase(size)} {...p}>
      <path d="M12 3v6M12 15v6M3 12h6M15 12h6M5.5 5.5l4 4M14.5 14.5l4 4M5.5 18.5l4-4M14.5 9.5l4-4" />
    </svg>
  );
}

export function MoonIcon({ size = 16, ...p }: IconProps) {
  return (
    <svg {...svgBase(size)} {...p}>
      <path d="M21 12.79A9 9 0 1111.21 3 7 7 0 0021 12.79z" />
    </svg>
  );
}

export function SunIcon({ size = 16, ...p }: IconProps) {
  return (
    <svg {...svgBase(size)} {...p}>
      <circle cx="12" cy="12" r="5" />
      <path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42" />
    </svg>
  );
}

export function BellIcon({ size = 16, ...p }: IconProps) {
  return (
    <svg {...svgBase(size)} {...p}>
      <path d="M18 8A6 6 0 006 8c0 7-3 9-3 9h18s-3-2-3-9" />
      <path d="M13.73 21a2 2 0 01-3.46 0" />
    </svg>
  );
}

export function SearchIcon({ size = 16, ...p }: IconProps) {
  return (
    <svg {...svgBase(size)} {...p}>
      <circle cx="11" cy="11" r="8" />
      <path d="M21 21l-4.35-4.35" />
    </svg>
  );
}

export const Icon = {
  shield: ShieldIcon,
  camera: CameraIcon,
  mail: MailIcon,
  check: CheckIcon,
  x: XIcon,
  warn: WarnIcon,
  arrowRight: ArrowRightIcon,
  cpu: CpuIcon,
  layers: LayersIcon,
  refresh: RefreshIcon,
  spark: SparkIcon,
  moon: MoonIcon,
  sun: SunIcon,
  bell: BellIcon,
  search: SearchIcon,
};
