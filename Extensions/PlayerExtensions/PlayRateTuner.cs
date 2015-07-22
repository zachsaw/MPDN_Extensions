// This file is a part of MPDN Extensions.
// https://github.com/zachsaw/MPDN_Extensions
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 3.0 of the License, or (at your option) any later version.
// 
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
// 
// You should have received a copy of the GNU Lesser General Public
// License along with this library.
// 
using System;
using System.Collections.Generic;
using System.Linq;
using Mpdn.Extensions.Framework;

namespace Mpdn.Extensions.PlayerExtensions
{
    public class PlayRateTuner : PlayerExtension<PlayRateTunerSettings, PlayRateTunerConfigDialog>
    {
        public override ExtensionUiDescriptor Descriptor
        {
            get
            {
                return new ExtensionUiDescriptor
                {
                    Guid = new Guid("DEAB2614-C54A-4A2B-9675-D133B002D1DC"),
                    Name = "Rate Tuner",
                    Description = "Changes playback rate to match display refresh rate"
                };
            }
        }

        protected override string ConfigFileName
        {
            get { return "RateTuner"; }
        }

        public override void Initialize()
        {
            base.Initialize();
            Media.Loaded += MediaLoaded;
        }

        public override void Destroy()
        {
            Media.Loaded -= MediaLoaded;
            base.Destroy();
        }

        private void MediaLoaded(object sender, EventArgs eventArgs)
        {
            Player.Playback.BaseRate = 1.0;

            if (!Settings.Activate)
                return;

            var tuning = Settings.Tunings.FirstOrDefault(t => VideoSpecifier.Match(t.Specifier));
            Player.Playback.BaseRate = tuning == null ? 1.0 : tuning.Rate;
        }
    }

    public class PlayRateTunerSettings
    {
        public PlayRateTunerSettings()
        {
            Activate = false;
            Tunings = new List<Tuning>();
        }

        public bool Activate { get; set; }
        public List<Tuning> Tunings { get; set; }

        public class Tuning
        {
            public string Specifier { get; private set; }
            public double Rate { get; private set; }

            // Do not remove this parameterless ctor (it is required by YAX for deserialisation)!
            public Tuning()
            {
            }

            public Tuning(string specifier, double rate)
            {
                Specifier = specifier;
                Rate = rate;
            }
        }
    }
}
